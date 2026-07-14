# Generalisation-Audit Runbook

A reusable methodology for auditing the LaSSI pipeline: (1) reverify it is in a known-good, behaviour-stable state, and (2) hunt for newly-introduced duplication, hardcoded word-lists, and one-off rules that should be expressed as general, data-driven mechanisms. Run cold, on demand, or at intervals after other work lands. Read `CLAUDE.md` first — it is the source of truth for architecture; this runbook adds the audit methodology and validation discipline.

## Core philosophy

The pipeline applies grammar rules to an inherently-flawed dependency tree, so it relies on many rules/vocabularies to repair structures and compare meaning. The project direction is: **express that knowledge as DATA in a single source of truth, interpreted by a small reusable engine — not as scattered Python branches, word-lists, or one-function-per-case.** Overfitting to `neet/evidence_cases` is the failure mode to fight.

Five patterns to enforce (each has a worked precedent in the codebase):

1. **Vocabulary → ontology classes, never Python word-lists.** Membership lives in HOnK (`HOnK.ttl`) or `raw_data/**` files, exposed by a getter, resolved through `KernelOntologyMatchers.matches_class`. Precedent: `OccurrenceVerb`, `RelativePronoun`, `Copula`, `EventClassifierHeadNoun`.
2. **Rules → declarative `premise → consequence` data.** Structural rewrites live in `raw_data/structural_rewrites.json`; the engine is `LaSSI/ner/structural_rewrites/declarative.py`. A new rule should be a JSON edit; `impl: "module:Class"` is the labelled escape hatch, the exception not the norm.
3. **Classification/semantics → derived from existing metadata.** Precedent: the ex-post "distinguishing keys" derive from `logical_analysis.json`'s `attachTo` plus a small `similarity_semantics` block — not a hand-maintained list. Before adding a classification list, ask "is this already implied by metadata we have?"
4. **Single registry drives derived artefacts.** Precedent: `HOnK._LOOKUP_SET_REGISTRY` drives inits/clears/getters; `structural_rewrites.json`'s `rules` array is the only rule registry.
5. **Shared primitives, not copy-paste.** Helpers appearing 2–3× belong in one home (`base.py`, `predicates.py`, `consequence_primitives.py`).

The bar for action: **can this be expressed via an existing data/ontology mechanism without bespoke Python?** If yes, generalise. If genuinely irreducible (needs the dependency graph, multi-pass surgery, or a real semantic judgment), leave it as labelled Python and say so — data-shaped code that hides a one-off behind a name re-creates the anti-pattern. (The 2026-06 audit concluded the remaining `impl:` share is irreducible structural surgery; don't force-convert it.)

## Validation discipline (non-negotiable)

- Interpreter: `~/PycharmProjects/LaSSI/.venv/bin/python`; **run from the repo root** (stale site-packages snapshot otherwise shadows the repo — verify with `python -c "import LaSSI, os; print(os.path.dirname(LaSSI.__file__))"`).
- Two gates:
  - Fast, no DB: `python -m unittest LaSSI/tests/test_structural_rewrites.py` (fake HOnK/matchers).
  - Source of truth: `python -m unittest LaSSI/tests/test_similarities.py` — compares `catabolites/<case>/matrices/confusion_matrices_Logical.json` against `LaSSI/tests/assertions/similarities_neet.json`. Baseline is all-Completely-Working (currently 20/20). `"null"` in the gold = any value strictly in (0,1). Do **not** trust `test_assertions.test_string_reps` (frozenset-ordering-dependent).
- **The cache gotcha:** edits to the comparison/similarity engine (`ModelSearch` / `ConstituentComparator` / `TabularCWASemantics`) are no-ops unless you delete `catabolites/<case>/SentenceRepresentation.Logical/` and `catabolites/<case>/matrices/confusion_matrices_Logical.json`, then recompute with `eval/_recompute_matrices.py <case>…`. Structural-rewrite/eFOL changes are *not* affected by this cache — an identical matrix after regeneration is your behaviour-neutrality proof.
- meuDB is the slow stage; default `delete_files(target_folders=[...])` preserves it. Re-run many cases in ONE Python process (HOnK loads once) — `regen_gold.sh` / `gold_driver.py`.
- **Behaviour-neutral refactors must prove it:** regenerate affected cases and diff matrices (`eval/diff_matrices.py`). If a value changes, justify it logically and only then update `similarities_neet.json` (gold is directional: cell `[i][j]` = `P(Sj | Si)`, and may be corrected when a computed value is more sound) — or revert.
- **Attribute regressions precisely:** revert your edit and re-run (cache cleared) to distinguish your change from pre-existing/WIP breakage. Know which layer you're editing — the verdict path is `ExpandConstituents.compare → ModelSearch._guard_contextual_implication`; the matrix cell is the BDD ratio from `get_straightforward_id_similarity` downstream of it.

## The hunt — smells to grep for

- **A. Hardcoded vocabulary** duplicating HOnK/raw_data: literal word sets (`lower() in {`, frozenset literals of domain words, `== "be"` checks). Grep `HOnK.ttl` for an existing class first (grep, never read — it's huge); else `raw_data/<verbs|nouns|…>/<file>.txt` + `_LOOKUP_SET_REGISTRY`/`_type_lookup_map` + `matches_class` `class_getters`.
- **B. Duplicated helpers** across modules (copula/freeze/lemmatise/recursive-walk shapes) → consolidate into the shared home, delete copies.
- **C. One-off rule/opcode-per-rule patterns** in the rewrite layer: a premise/consequence name 1:1 with a single rule. Prefer composition from the general ops (`Move`/`Append`/`ReplaceSlot`/`Drop`/`PromoteToTarget`/`Flatten`); new primitives only when irreducible, in `consequence_primitives.py`, generally named.
- **D. Classification lists that could be derived** from `logical_analysis.json`, `dependency_roles.json`, or an existing HOnK getter.
- **E. Multiply-listed registries** (a list + `__all__` + import block; a set + init + getter) → collapse to one registry that derives the rest.
- **F. Lexical sets drifted into Python or JSON config that are actually ontology** (NER-tag/preposition lists are legitimately data; verb/noun vocabulary is not).
- **G. Dead code / stale references**: unused imports, never-firing rules (trace with `_audit_trace_one.py`, which monkeypatches `RuleRegistry.apply_phase` and writes `_rule_firing.json` per case), comments naming deleted classes.

## Output format

Report: **Reverify** (gate results vs baseline, attribution if regressed) · **Findings** (file:line, smell A–G, reducible/irreducible verdict, proposed/applied generalisation) · **Applied changes** (with behaviour-neutral proof) · **Deferred/needs-owner-decision** (anything semantic — which keys are distinguishing, which verbs synonymous — don't guess) · **Memory** (durable new gotchas → project memory + `CLAUDE.md`).

Bias toward fewer, well-justified, behaviour-neutral generalisations over sweeping rewrites. The goal: adding a new evidence case means editing data (ontology/JSON), not Python.
