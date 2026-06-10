# <img src="extra/lassiLogo.svg" style="height:80px; width: auto;" alt="Logo: Credits to Oliver Robert Fox (2024)" /> LaSSI

LaSSI stands for `LogicAl, Structural, and Semantic text Interpretation`. Rather than relying solely on learned representations, LaSSI interprets natural language through a pipeline of verifiable, symbolic steps; converting raw text into Extended First-Order Logic (eFOL) formulae and computing inter-sentence similarity via BDD model-counting. This design keeps every reasoning decision auditable and traceable, in the spirit of Verified Artificial Intelligence. The pipeline is domain-general; current experiments use the **NEET** corpus of transport, roadworks, and crime notices as an evaluation benchmark.

## Authors

* Oliver Robert Fox (2023 –)
* Giacomo Bergami (2020 – 2025)
* Franco O. Saez Vander Linder (2025)

---

## Pipeline Architecture

Data flows through the following phases (intermediate results cached per-sentence in `catabolites/`):

| Phase | Module | Description                                                                                                                          |
|---|---|--------------------------------------------------------------------------------------------------------------------------------------|
| SentenceLoader | `phases/SentenceLoader.py` | Ingests YAML rows                                                                                                                    |
| StructuredSentenceLoader | `phases/StructuredSentenceLoader.py` | Splits rows into structural chunks; protects ISO timestamps, date ranges, and metrics                                                |
| RowChunkPipeline | `phases/RowChunkPipeline.py` | Classifies chunk roles (ACTION / PROSE / HEADER / TOPIC_HEADER / STATUS / …), merges per-chunk kernels back to row level             |
| ExplainTextWithNER | `phases/ResolveBasicTypes.py` | Stanza POS tagging and dependency parsing; fuzzy matching against HOnK to produce a Multi-Entity Unit Database (meuDB)               |
| GetGSMString | `phases/GetGSMString.py` | Calls Java/Stanford CoreNLP to produce a Generalised Semistructured Model (GSM)                                                      |
| ApplyGraphGrammars | `phases/ApplyGraphGrammars.py` | Runs DatagramDB graph-grammar rules                                                                                                  |
| SemanticGraphRewriting | `phases/SemanticGraphRewriting.py` | Converts GSM to an internal EntityRelationship graph; applies structural rewrites                                                    |
| LogicalRewriting | `phases/LogicalRewriting.py` | Produces Extended FOL formulae                                                                                                       |
| CalculateMatrix | `phases/CalculateMatrix.py` | Computes the inter-sentence similarity matrix (eFOL BDD model-counting for `Logical` mode, or embedding/LLM backends for `FullText`) |

---

## Key Capabilities

- **Declarative structural-rewrite framework** — 39 rules (19 declarative JSON + 20 Python) that repair dependency-tree artefacts into clean eFOL kernels. The rule registry lives in `raw_data/structural_rewrites.json`; Python subclasses are reserved for rewrites requiring genuine graph surgery.

- **eFOL BDD similarity** — directional entailment score P(Sj | Si) computed via BDD model-counting with a distinguishing-content cap. Cap conditions and construct metadata are data-driven from `raw_data/logical_analysis.json` — no hand-maintained lists in code.

- **HOnK ontology** — a local RDF knowledge graph (`HOnK.ttl`) providing synonym/hypernym lookup at inference time via BFS over pre-built adjacency dicts. Extended with:
  - `LifecycleStates.ttl` — domain contradiction axioms (e.g. *reduction in spaces* ⊥ *operating normally*)
  - `Paraphrase.ttl` — predicate equivalences (e.g. *operate* ≡ *run*)

- **Multiple similarity backends** — selectable via the `transformer` constructor parameter:

  | Prefix | Backend | Example |
  |---|---|---|
  | *(none)* | eFOL BDD model-counting | default `Logical` mode |
  | bare string | HuggingFace sentence-transformer | `all-MiniLM-L6-v2` |
  | `NLI#` | Transformer NLI entailment | `NLI#cross-encoder/nli-MiniLM2-L6-H768` |
  | `LLM#` | Local Ollama LLM | `LLM#qwen2.5:7b` |
  | `LLMHOnK#` | LLM grounded in HOnK facts | `LLMHOnK#qwen2.5:7b#all` |
  | `RAG#` | ColBERT retrieval-augmented | `RAG#colbert-ir/colbertv2.0` |

- **Explainer dashboard** — Flask/Plotly web interface for inspecting confusion matrices, logical forms, and per-sentence reasoning provenance.

- **Evaluation tooling** — `eval/` contains an ablation runner for the NEET 3-class claim-verification task (Supported / Refuted / Not Enough Evidence) comparing ungrounded LLMs against HOnK-grounded variants.

---

## Installing

Install the required system dependencies first:

```bash
pip install requests==2.31.0
pip install setuptools==66.1.1
sudo apt-get install libpq-dev   # Linux; macOS: brew install libpq
```

Then install the package:

```bash
pip install .
```

Also ensure **Java** is installed (required for Stanford CoreNLP).

### PostgreSQL

```bash
sudo apt install postgresql -y
sudo -u postgres psql
```
```postgresql
create database lassi;
create user lassi with encrypted password 'drowssap';
grant all privileges on database lassi to lassi;
\c lassi postgres
grant all on schema public to lassi;
\q
```

Database connection (PostgreSQL credentials and fuzzy-DB dataset URLs for ConceptNet and GeoNames) is configured in `connection.yaml`.

---

## Running

```bash
# Default run
python main.py

# Custom dataset and config
python main.py <dataset.yaml> <connection.yaml>
```

The `SentenceRepresentation` enum in `LaSSI/Configuration.py` controls output format and similarity mode:

| Value | Description |
|---|---|
| `FullText` | Raw text passthrough; embedding/LLM similarity |
| `SimpleGraph` | Graph representation |
| `LogicalGraph` | Extended FOL + graph |
| `Logical` | Extended FOL formulae; eFOL BDD similarity (default) |
| `SimpleGraphDisabledAPriori` | `SimpleGraph` with a-priori similarity disabled |
| `LogicalGraphDisabledAPriori` | `LogicalGraph` with a-priori disabled |
| `LogicalDisabledAPriori` | `Logical` with a-priori disabled |

Intermediate representations are cached in `catabolites/` so subsequent runs skip completed phases.

---

## Tests

The pipeline must have been run at least once to populate `catabolites/` before running assertion-based tests.

```bash
# Structural-rewrite unit tests
python -m unittest LaSSI/tests/test_structural_rewrites.py

# Similarity gold tests (source of truth)
python -m unittest LaSSI/tests/test_similarities.py

# Benchmarks
python LaSSI/tests/benchmark_sentences.py
python LaSSI/tests/run_all_sentences.py
```

`test_similarities.py` compares per-case confusion matrices against `LaSSI/tests/assertions/similarities_neet.json`. Gold entries marked `"null"` accept any score strictly between 0.0 and 1.0 (partial implication); concrete values must match exactly. Benchmark results are written to `catabolites/benchmark.csv` and `results/`.

---

## Datasets

The primary evaluation corpus is NEET — a collection of transport, roadworks, and crime notices annotated for claim verification. The annotated dataset is available on [OSF](https://osf.io/fu4bg/overview?view_only=c321084ba74c488fb587fc4b54cc290f).

Test sentences are in `test_sentences/`; benchmarking sentences are in `test_sentences/benchmarking/`.

---

## Future Work

- Extend structural-rewrite coverage to more complex sentence structures (multi-clause, relative-clause chains)
- Improve robustness of telegraphic headline parsing (pseudo-verb recovery edge cases)
- Increase domain coverage beyond current evaluation datasets
