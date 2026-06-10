# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**LaSSI** (LogicAl, Structural, and Semantic text Interpretation) is a Python NLP pipeline that interprets natural language through logical, verifiable steps rather than purely learning-based approaches. It converts raw text into Extended First-Order Logic (FOL) formulas via a multi-phase pipeline backed by PostgreSQL, Java/Stanford CoreNLP, and HuggingFace transformers.

The pipeline is general-purpose, though current experiments focus on the **NEET** corpus (roadworks and transport notices found under `neet/`).

## Python Interpreter

Always use the project venv:

```bash
~/PycharmProjects/LaSSI/.venv/bin/python
```

## Installation

```bash
# Prerequisites
pip install requests==2.31.0 setuptools==66.1.1
sudo apt-get install libpq-dev   # Linux; on macOS use Homebrew equivalent
# Java must also be installed

# Install package
pip install .
```

**PostgreSQL setup (one-time):**
```bash
sudo apt install postgresql -y
sudo -u postgres psql
```
```sql
create database conceptnet;
create user lassi with encrypted password 'drowssap';
grant all privileges on database conceptnet to lassi;
\c conceptnet postgres
grant all on schema public to lassi;
```

Database connection is configured via `connection.yaml` (PostgreSQL credentials + fuzzy DB dataset URLs for ConceptNet and GeoNames).

## Running

```bash
# Default run (uses neet/evidence_cases/crime_006.yaml + connection.yaml)
python main.py

# Custom dataset and config
python main.py <dataset.yaml> <connection.yaml>
```

The `SentenceRepresentation` enum in `LaSSI/Configuration.py` controls the output format:

| Value | Description |
|---|---|
| `Logical` | Extended FOL formulae (default) |
| `LogicalGraph` | FOL + graph |
| `SimpleGraph` | Graph only |
| `FullText` | Raw text passthrough |
| `LogicalDisabledAPriori` | `Logical` with a-priori similarity disabled |
| `LogicalGraphDisabledAPriori` | `LogicalGraph` with a-priori disabled |
| `SimpleGraphDisabledAPriori` | `SimpleGraph` with a-priori disabled |

`main_remote.py` exists as an alternative entry point that connects LaSSI to a remotely-deployed HOnK service (`HOnKRemoteSingleton`). It is currently unused in favour of loading the local `HOnK.ttl` directly.

## Tests

Tests compare pipeline output against gold-standard string representations stored in `LaSSI/tests/assertions/assertions_neet.txt`. The test reads intermediate outputs from the `catabolites/` directory, so **the pipeline must have been run first** to populate `catabolites/`.

```bash
# Run the main assertion suite
~/PycharmProjects/LaSSI/.venv/bin/python -m unittest LaSSI/tests/test_assertions.py

# Run a single test method
~/PycharmProjects/LaSSI/.venv/bin/python -m unittest LaSSI.tests.test_assertions.TestLaSSI.test_string_reps

# Run similarity tests
~/PycharmProjects/LaSSI/.venv/bin/python -m unittest LaSSI/tests/test_similarities.py

# Run structural-rewrite unit tests (no pipeline run required)
~/PycharmProjects/LaSSI/.venv/bin/python -m unittest LaSSI/tests/test_structural_rewrites.py

# Run benchmarks
~/PycharmProjects/LaSSI/.venv/bin/python LaSSI/tests/benchmark_sentences.py
~/PycharmProjects/LaSSI/.venv/bin/python LaSSI/tests/run_all_sentences.py
```

Benchmark results are written to `catabolites/benchmark.csv` and `results/`.

### Similarity tests are the source of truth (not string-reps)

`LaSSI/tests/test_similarities.py` compares each case's `catabolites/<case>/matrices/confusion_matrices_Logical.json` against `LaSSI/tests/assertions/similarities_neet.json`. (All `confusion_matrices_*` files live under the per-case `matrices/` subdirectory; `LaSSI.py` auto-migrates old top-level files on the next run for that case.) Only cases with an entry there are checked (others skipped); baseline is all-Completely-Working (currently 20/20). In the gold, **`"null"` means "any value strictly between 0.0 and 1.0"** (a partial/implication score); concrete values must match exactly. The gold matrices are **directional/asymmetric** (cell `[i][j]` = `P(Sj | Si)`), and **may be changed if a computed value is more logically sound** — judge each disagreement on its merits.

Regeneration/verification tooling: `zsh regen_gold.sh` (full per-case rebuild of the 20 gold cases; `regen_gold.sh all` = all 30; or pass explicit case names — one fresh Python process per case with a repo-root `sys.path` guard); `eval/_recompute_matrices.py case…` (cheap comparison-only recompute in ONE process after deleting `SentenceRepresentation.Logical` + the matrix — for comparison-engine edits); `eval/diff_matrices.py <snapshot_dir> [case…]` (cell-level matrix diff against a snapshot directory of `<case>.json` files, printing old/new/gold + both sentence texts).

**Prefer the similarity suite over `test_assertions.test_string_reps`.** String-reps are frozenset-backed, so property-bag *ordering* differs run-to-run even when the matrix is byte-identical — `test_string_reps` is unreliable. Sentences with no assertion line in `assertions_neet.txt` are known-broken cases.

## Pipeline Architecture

Data flows through these phases in `LaSSI/LaSSI.py`:

```
YAML Input
  → SentenceLoader          (LaSSI/phases/SentenceLoader.py)
  → StructuredSentenceLoader (LaSSI/phases/StructuredSentenceLoader.py)
       • Splits scraped notice rows into structural chunks before CoreNLP
       • Protects atomic spans (ISO timestamps, date ranges, metrics)
       • Produces StructuredChunk objects with delimiter and label metadata
  → RowChunkPipeline        (LaSSI/phases/RowChunkPipeline.py)
       • Expands rows into sub-sentences, profiles each chunk's role
         (ACTION / PROSE / HEADER / TOPIC_HEADER / REPORT_HEADER /
          ATTRIBUTE / STATUS / TIME_RANGE / CONTEXT via ChunkProfiler)
       • Merges per-chunk kernels back into one kernel per YAML row
       • Wraps ACTION runs in SetOfSingletons(AND); synthesises
         be(topic, AND(actions)) when a TOPIC_HEADER anchors the row
  → ExplainTextWithNER      (LaSSI/phases/ResolveBasicTypes.py)
       • Stanza POS tagging & dependency parsing
       • Fuzzy string matching against Parmenides ontology, GeoNames, ConceptNet5
       • Produces MeuDB (Multi-Entity Unit Database) per sentence
  → GetGSMString            (LaSSI/phases/GetGSMString.py)
       • Calls Java/Stanford CoreNLP service to produce Graph String Model (GSM)
  → ApplyGraphGrammars      (LaSSI/phases/ApplyGraphGrammars.py)
       • Runs DatagramDB graph grammar rules from LaSSI/resources/gsm_query.txt
  → SemanticGraphRewriting  (LaSSI/phases/SemanticGraphRewriting.py)
       • Converts GSM to internal EntityRelationship graph
       • Applies structural rewrites (declarative rules from raw_data/structural_rewrites.json)
  → LogicalRewriting        (LaSSI/phases/LogicalRewriting.py)
       • Produces Extended FOL formulae
  → CalculateMatrix         (LaSSI/phases/CalculateMatrix.py)
       • Computes the inter-sentence similarity matrix. For
         SentenceRepresentation.Logical this is eFOL BDD model-counting
         (LaSSI/structures/extended_fol/TabularCWASemantics.py), NOT embeddings;
         embeddings / ColBERT RAG are used for the FullText representation.
         See "Ex-Post Similarity (eFOL semantics)" below.
```

Intermediate representations are cached in `catabolites/` as JSON/pickle files, enabling re-runs to skip completed phases.

## Key Modules

- **`LaSSI/LaSSI.py`** — Main orchestrator. Key constructor parameters:
  - `transformer` — sentence-transformer model string (see Transformer strings below)
  - `disable_a_priori` — skip a-priori (pre-similarity) phase
  - `disable_fuzzy_honk` — skip loading the local HOnK TTL and PostgreSQL services
  - `run_ex_post` — run ex-post (post-logical) similarity phase (default `True`)
  - `useId` — use node IDs rather than labels in output
  - `generate_png_matrix` — write a PNG similarity matrix to `catabolites/`
  - `use_multiprocessing` — spawn up to 8 workers via `ProcessPoolExecutor` (default `True`)
- **`LaSSI/Configuration.py`** — `SentenceRepresentation` enum
- **`LaSSI/phases/StructuredSentenceLoader.py`** — Pre-parser: splits raw notice rows into `StructuredChunk` objects without assigning roles
- **`LaSSI/phases/RowChunkPipeline.py`** — Orchestrates row→chunk→row expansion/merge; owns chunk metadata and the synthetic-node counter
- **`LaSSI/phases/ResolveSingleSentence.py`** — Per-sentence worker (handles ISO-8601 datetime MEU injection and delegates to Stanza)
- **`LaSSI/ner/ChunkProfiler.py`** — `ChunkRole` enum + `profile()` function; classifies chunks using HOnK vocabulary
- **`LaSSI/external_services/Services.py`** — Singleton that initialises all external NLP/DB services (Stanza, FuzzyStringMatchDatabase, StanzaNLPExtractor)
- **`LaSSI/Parmenides/`** — RDF ontology backed by rdflib-sqlalchemy; handles TBox reasoning, WordNet integration, and typed object database
- **`LaSSI/HOnK/HOnK.py`** — Ontology graph loaded from `HOnK.ttl`; provides `getSynonymy`, `getSuperTypes`, `getSubTypes` via BFS over pre-built adjacency dicts (not live SPARQL)
- **`LaSSI/HOnK/TBox/`** — TBox reasoning components (the eFOL comparison engine):
  - `ExpandConstituents.py` — drives pairwise comparison; `compare()` delegates to `ModelSearch`
  - `ConstituentComparator.py` — per-constituent comparison + property compatibility; produces `CasusHappening` verdicts
  - `LifecycleManager.py` — lifecycle-state comparisons (e.g. "no suspect" vs "arrested")
  - `ParaphraseManager.py` — paraphrase/concept-of matching (`Paraphrase.ttl`)
  - `SpatialReasoner.py` — geographic and spatial contradiction detection
- **`LaSSI/structures/extended_fol/`** — eFOL + similarity:
  - `Formulae.py` — FOL formula types (`FUnaryPredicate`/`FBinaryPredicate` carry `.properties`; `FNot`/`FAnd`/`FOr` wrap `.arg`/`.args`)
  - `ModelSearch.py` — `compare()` + `_guard_contextual_implication()`: directed `CasusHappening` verdict per pair. Owns `_KERNEL_LOGICAL_CONTEXT_KEYS` (≈ `attachTo:Kernel` keys, minus SPACE/TIME) and `_KERNEL_KEY_EQUIVALENCES` (paraphrastic key aliases)
  - `TabularCWASemantics.py` — builds the float similarity matrix via BDD model-counting (see "Ex-Post Similarity")
- **`LaSSI/explainer/`** — `FullExplainer`, `LaSSIExplainer`, `ReportBuilder`: produce human-readable explanations and reports from pipeline provenance
- **`LaSSI/structures/`** — Core data models:
  - `meuDB/meuDB.py` — Multi-entity unit entries with types and confidence scores
  - `internal_graph/` — EntityRelationship graph (nodes/edges)
  - `extended_fol/Formulae.py` — FOL formula types (Predicates, Variables, Quantifiers)
  - `provenance/GraphProvenance.py` — Traces reasoning decisions
- **`LaSSI/similarities/`** — Similarity scoring; supports HuggingFace sentence-transformers, ColBERT RAG (`RAG#colbert-ir/colbertv2.0`), and implication classifiers (`Log#<hf-model-id>`)
- **`LaSSI/utils/configurations.py`** — `LegacySemanticConfiguration` (transformer model, similarity thresholds)
- **`raw_data/structural_rewrites.json`** — Single source of truth for declarative structural-rewrite rules and lexical sets (see below)

## GSM Grammar (`gsm_query.txt`) vs Python — the layer boundary

`LaSSI/resources/gsm_query.txt` (DatagramDB graph-grammar rules; syntax: https://github.com/LogDS/datagram-db/wiki/Syntax) runs over the raw CoreNLP dependency graph **before** any ontology/NER is attached: rules can test structure, `xpos`, `lemma` and dependency labels, but NOT HOnK classes or entity types. So in principle: *pure-structure parse repairs belong in GSM; ontology/NER-gated repairs belong in Python* (GraphBuilder pre-build mutations, kernel building, structural rewrites).

**However — empirical findings (2026-06 audit; reproduce with `eval/_gsm_probe.py <case> [query_file]`, which runs any query over a cached `catabolites/<case>/gsmDB.txt` and matches the pipeline's DatagramDB stage byte-identically):**
- **New rules appended to the existing grammar do not fire.** Two well-formed rules (a marker-chain property materialisation and an acl-participle re-point) each work when run as the *only* rule in the file, but are completely inert inside the full grammar, whether placed first or last — existing rules starve new ones of morphisms. Extending the grammar's behaviour requires re-carving existing rules' patterns/guards (corpus-wide blast radius, full rebuild per iteration), or engine-level investigation in the upstream datagram-db project.
- **Rule consequences have interface-replacement semantics**: the returned node `(X)` replaces the matched region in surrounding context (e.g. an `obl` edge to the matched source gets re-pointed to the returned node), and a multi-content containment (one `phi` label with several children) binds pattern variables non-obviously. Test every rule change empirically with the probe before trusting it.
- **The p3/p4 kernel-building family fires on only ~13 nodes across 8 of the 30 evidence cases** — on this corpus the dependency graph reaches the Python kernel builder largely raw. The Python layers are interpretation of well-formed parses at least as often as they are repairs of broken ones.

Practical rule: verify any `gsm_query.txt` edit with `eval/_gsm_probe.py` against the cached `gsmDB.txt` of the cases it should (and should not) affect, then a full per-case rebuild (`regen_gold.sh <case>`) — GSM edits invalidate `datagramdb_output.json` and everything downstream.

## Structural Rewrite Rules

Structural rewrites repair the (inherently flawed) dependency-tree-derived kernels into clean eFOL. They are **data-driven**, in the same `premise → consequence` shape as `logical_analysis.json`'s `derivation_rules`.

**Single source of truth: `raw_data/structural_rewrites.json`.** Two top-level keys:
- `lexical_sets` — a few NER-tag / preposition lists that are genuinely data, not ontology classes (e.g. `context_entity_types`). Spatial-relation labels live in `logical_analysis.json`'s `spatial_relations` block, not here.
- `rules` — the **complete ordered rule registry** (array order = within-phase application order; 30 rules as of the 2026-06 audit — 9 never-firing rules were deleted). `LaSSI/ner/structural_rewrites/__init__.py:default_registry()` builds it via `load_rules()`; **`__init__.py` contains no hardcoded rule list.**

Each `rules` entry is either:
- **Declarative** `{name, phase, premise, consequence}` — interpreted by `DeclarativeStructuralRewriteRule`. A `premise` maps reusable predicate names to value lists (ALL premises must hold; each ORs over its values); a `consequence` is an ordered list of reusable ops.
- **`impl: "module:ClassName"`** — a Python `StructuralRewriteRule` subclass, only for rewrites too bespoke to express as data (heavy graph traversal / multi-pass surgery, e.g. `auxiliary_periphrasis_promotion`, `participial_collapse`, `passive_progressive`). Prefer declarative.

**Module layout** (`LaSSI/ner/structural_rewrites/`):
- `declarative.py` — the **engine only**: premise/consequence dispatch + the general ops (`Move`, `Append`, `ReplaceSlot`, `Drop`, `PromoteToTarget`, `Flatten`) + `load_rules`/`load_declarative_rules`.
- `consequence_primitives.py` — the bespoke structural primitives ("escape hatch": `SplitModifier`, `MergeAdjacent`, `SwapHeadWithExtra`, …).
- `predicates.py` — shared class/structure predicates (`matches_class`, slot accessors, existential/AND constructors).
- `config.py` — loads the JSON. `base.py` — shared kernel surgery (`walk_kernel`, `replace_kernel`, `is_copula_surface`, `freeze_props`, `lemmatise_verb_phrase`).

**Premise vocabulary** (small, reusable; see `declarative._eval_premise`): `EdgeMatchedBy`/`SourceMatchedBy`/`TargetMatchedBy` (HOnK class), `SlotIsType`, `HasProperty`, `PropertyValueCount`, `PropertyMatchedBy` (`KEY:Class`), `NodePropertyMatchedBy`, `PropertyHasPromotableContent`, plus escape-hatch predicates.

**Vocabulary classes resolve through `KernelOntologyMatchers.matches_class`** against HOnK getters / `raw_data` files — **never hardcode word lists in Python.** To add a class (e.g. `OccurrenceVerb`): **first grep `HOnK.ttl` for an existing class**; else add `raw_data/<verbs|nouns>/<file>.txt`, register it in `HOnK._type_lookup_map`/`_load_support_lookup_sets` with a getter, and add the class to `matches_class`'s `class_getters`.

**To add/modify a rule: edit `raw_data/structural_rewrites.json`** — only write Python when the rewrite needs a genuinely new graph/structural primitive (add it to `consequence_primitives.py`, wire a named op in `declarative._apply_consequence`). `LaSSI/tests/test_structural_rewrites.py` runs rules in isolation (fake HOnK/matchers), no pipeline run needed.

## Ex-Post Similarity (eFOL semantics)

For `SentenceRepresentation.Logical`, the similarity matrix is computed from the eFOL formulae by **BDD model-counting**, not embeddings (`CalculateMatrix` → `LaSSI.py:_calculate_matrix`).

**Comparison stack:** `ExpandConstituents.compare()` → `ModelSearch.compare()`/`_guard_contextual_implication()` → per-pair `CasusHappening` verdict (EQUIVALENT / implication / INDIFFERENT / EXCLUSIVES), using `ConstituentComparator` for per-constituent property compatibility. The float matrix is then built in `TabularCWASemantics`: `__call__(i,j)` → `get_straightforward_id_similarity(i,j)` returns the **directional entailment ratio `P(Sj | Si)`** via BDD counting (`_raw_id_similarity`). The cell is that single directional number (1.0 = entails, 0.0 = exclusive, in-between = partial).

**Distinguishing-content cap** (`get_straightforward_id_similarity`): logical-context / SPECIFICATION properties are *not* BDD variables, so a more-specific sentence would spuriously score 1.0 against a genuinely different one. The cap reduces `i→j` to a partial score when, for that direction: **(A)** the *target* asserts a distinguishing key the source lacks; **(S)** they differ on an *identity-refining* key (symmetric); or **(B)** different surface relations carry different distinguishing content. A more-specific sentence entailing a less-specific one under the *same* relation stays 1.0.

**The distinguishing-key classification is DATA-DRIVEN from `raw_data/logical_analysis.json`** (`TabularCWASemantics._similarity_key_sets`, cached):
- `types[*].attachTo` — `Kernel` constructs (event clauses) are distinguishing-*directional*; `Singleton` constructs are argument facets.
- A `similarity_semantics` block declares the two judgments `attachTo` can't encode: `circumstantial` (excluded — where/when/status/purpose framing) and `identity_refining` (symmetric-distinguishing, e.g. `specification`).
- Derived: `distinguishing = (Kernel constructs − circumstantial) ∪ identity_refining`.
- **A new `attachTo: Kernel` construct becomes distinguishing automatically; otherwise edit one of the two lists in `logical_analysis.json` — never `TabularCWASemantics`.**

`get_explained_id_similarity` (DataFrame-based) feeds `pairwise_truth_tables.json` (explanations only), **not** the matrix.

**ModelSearch's guard sets are likewise derived, never hand-edited.** `LaSSI/utils/logical_analysis_reader.py` (cached, dependency-free) derives from `logical_analysis.json`: `kernel_context_keys()` = `attachTo:Kernel` constructs − `similarity_semantics.kernel_context_excluded` (space/time) + `key_spelling_aliases`; `kernel_context_monotonicity()` partitions those by each type-spec's `monotonicity` field — `restrictive` (default: more-specific ⇒ less-specific; the guard blocks an implication when the RHS asserts a key the LHS lacks) vs `intensional` (MODALITY: non-veridical operator; modal ⇏ factual, so the guard blocks when the LHS carries the key and the RHS doesn't); `paraphrastic_slots()` (slot aliases, e.g. AIM_OBJECTIVE↔TEMPORAL_CONTEXT); `spatial_relation_labels(subset)` (the `spatial_relations` block — single registry for "stay in place"/"near place"/… labels, consumed by `SpatialReasoner`, `KernelPostProcessor` and `participial_predicate_promotion`). To change any of these behaviours, edit `logical_analysis.json`.

`raw_data/logical_analysis.json` itself has four top-level keys: `types` (construct metadata: `attachTo` Kernel/Singleton, `argument`, optional `monotonicity`), `similarity_semantics` (above), `spatial_relations`, and `derivation_rules` (`premise → classification` rules that route prepositions/contexts to constructs, evaluated by `SemanticRoleRewriting`).

## Catabolites Caching & Per-Case Regeneration

`catabolites/<case>/` caches each phase so re-runs skip completed work. This matters a lot when testing changes:

- **The eFOL comparison cache is the #1 gotcha.** `catabolites/<case>/SentenceRepresentation.Logical/` holds the cached pairwise comparison results (`_ec.pickle`, `pairwise_truth_tables.json`, `explain_*.json`). **Editing the comparison/similarity engine (`ModelSearch` / `ConstituentComparator` / `TabularCWASemantics`) has NO effect unless this directory is deleted** — the cached verdicts are served instead, so your change silently no-ops. To force a real recompute:
  ```bash
  rm -rf catabolites/<case>/SentenceRepresentation.Logical \
         catabolites/<case>/matrices/confusion_matrices_Logical.json
  ```
  then recompute with `eval/_recompute_matrices.py <case>…` (one process, HOnK loads once). `LaSSI/tests/delete_catabolites.delete_files(target_folders=[...])` does **not** clear this dir unless you also pass a `transformation` arg. (Structural-rewrite / eFOL changes are unaffected by this cache — it is keyed on the regenerated eFOL, so a behaviour-neutral eFOL correctly yields an identical matrix.)
- **meuDB is the slow stage.** `delete_files(target_folders=[...])` (default `delete_all_files=False`) preserves `meuDBs.json` while clearing gsmDB / grammar / logical_rewriting / matrices — re-running regenerates from cached NER (fast). Pass `delete_all_files=True` only to force the slow meuDB rebuild.
- **Structural-rewrite changes have NO cheap recompute.** The structural rewrites (`LaSSI/ner/structural_rewrites/**`, `KernelPostProcessor`) run inside `SemanticGraphRewriting` → `internals.json`, which `delete_files` does *not* clear. To re-run them you must regenerate `internals.json`, and `_invalidate_stale_cache` couples that to a meuDB delete (missing internals ⇒ meuDB + gsmDB also deleted) — i.e. a full ~8 min/case rebuild. So verify structural-rewrite edits with `test_structural_rewrites.py` (fast, fake HOnK) + a targeted regen of only the case(s) that exercise the rule; reserve a full sweep for genuine doubt. Don't try to reuse a cached meuDB to skip NER — the per-row merge can leave it inconsistent with per-sub-sentence internals on chunked cases.
- **Re-run one case:** `python main.py neet/evidence_cases/<case>.yaml connection.yaml` writes `catabolites/<case>/`. With warm HOnK caches, the ontology loads in ~15–20s (not minutes).
- **Run from the repo root.** `pip install .` leaves a stale snapshot of `LaSSI` in site-packages; running from the repo root puts the repo first on `sys.path` and shadows it. Confirm with `python -c "import LaSSI, os; print(os.path.dirname(LaSSI.__file__))"` (must be the repo path). Scripts run from elsewhere (e.g. `/tmp`) import the stale snapshot.

## Important Conventions

- **Legacy/refactoring pattern**: Files suffixed with `X` (e.g., `FooX.py`) hold legacy versions kept for backward compatibility during refactors; prefer the non-`X` version.
- **Multiprocessing**: `use_multiprocessing=True` (default) spawns up to 8 workers via `ProcessPoolExecutor` in `ExplainTextWithNER`. Always use `if __name__ == '__main__':` guard when calling the pipeline.
- **Transformer strings**: Prefix `RAG#` for ColBERT models, `Log#` for implication classifiers, bare string for sentence-transformers.
- **`catabolites/` directory**: Runtime output directory created automatically. Each sentence gets a subdirectory with `string_rep.txt`, intermediate JSON, and graph files. Do not check these into git.
- **`connection.yaml`**: Required for fuzzy DB and PostgreSQL; structure follows `DatabaseConfiguration` from `LaSSI/external_services/utilities/DatabaseConfiguration.py`.
- **HOnK adjacency cache**: HOnK pre-builds relation-adjacency dicts at load time and caches them as a pickle (`~170 MB`). Cache freshness is anchored on `HOnK.ttl` mtime — if the TTL changes, delete the pickle to force a rebuild.
