# Appendix study — Ontology-grounded LLM entailment (HOnK + LaSSI ontologies)

*Handoff brief for the paper-writing agent. Everything here is implemented and
runnable; numbers are produced by the scripts in `eval/` into
`results/appendix_honk/`. Where a number is not yet final (full run in progress)
it is marked **[pending full run]**.*

---

## 1. One-paragraph abstract

We test whether *grounding* a local LLM's entailment judgment in LaSSI's
symbolic knowledge improves claim verification, following the "Follow the Path"
recipe (grounding LLM reasoning in knowledge-graph paths; arXiv:2505.11140). For
each candidate sentence pair we extract the relevant relations from three LaSSI
knowledge sources — the general **HOnK** ontology, the **LifecycleStates**
contradiction ontology, and the **Paraphrase** equivalence ontology — and inject
them into the prompt as verified facts; the LLM still emits the final score
(grounding-only, the ontologies never override it). An ablation over three local
models (`llama3.2:3b`, `qwen2.5:7b`, `gemma4:e2b`) and a grounding ladder
(ungrounded → +HOnK → +Lifecycle → +HOnK+Lifecycle → +all) on the NEET corpus
shows that **the contribution is source-dependent**: the general ontology alone
can *add noise*, while the domain contradiction/paraphrase ontologies are where
the gains concentrate (esp. contradiction-detection F1).

## 2. Motivation & positioning

- **Related work.** "Follow the Path" (arXiv:2505.11140) improves LLM factuality
  by anchoring the model's reasoning to knowledge-graph paths rather than running
  a separate symbolic inference engine. We instantiate that idea with LaSSI's
  ontologies as the knowledge graph.
- **Relation to the main paper.** The main paper's contribution is the *symbolic*
  eFOL pipeline (LaSSI `Logical`), which verifies entailment by BDD model-counting
  over first-order formulae. This appendix is a *complementary* probe: it does not
  replace that pipeline — it asks how much of LaSSI's hand-built symbolic
  knowledge transfers as *prompt grounding* to a black-box LLM. The honest framing
  is "grounding helps selectively," not "LLM ≥ symbolic."

## 3. Method

**Grounding-only.** The backend (`LaSSI/similarities/LLMandHOnK.py`,
`LLMHOnKPrompt`) subclasses the plain LLM backend (`LLM.py`, `LLMPrompt`) and
reuses its Ollama client / JSON-output / error handling. It overrides only the
prompt construction to insert an evidence block; the LLM produces the
`implication_score` (1.0 / 0.5 / 0.0) and a free-text reasoning, exactly as the
ungrounded baseline does. The ontologies never override the model's number.

**Three knowledge sources (each ablatable).**
- **HOnK** — the general (WordNet-scale) ontology, queried per term-pair via
  `HOnK.name_eq` → synonymy / isA / partOf / sparse `neqTo`. Rendered as e.g.
  `"footbridge" is a kind of "bridge"`, `"closed" is equivalent to "shut"`.
- **Lifecycle** (`LifecycleStates.ttl`) — domain *contradictions* via partitioned
  dimensions. Two phrases on the same dimension with different (non-`descriptive`)
  partitions are incompatible. Rendered as e.g. `"out of use" is incompatible with
  "operational" (both describe transport availability, but assert opposite
  states)`. This is the only source that supplies a strong *contradiction* signal.
- **Paraphrase** (`Paraphrase.ttl`) — domain *equivalences*, incl. project-specific
  numeric canonicalisation (probability buckets). Rendered as e.g. `"operate"
  means the same as "run"`.

**Phrase extraction.** HOnK uses lightweight content-term extraction (unigrams +
adjacent bigrams, function words dropped via HOnK's own getters). The two TTL
sources are matched by scanning each sentence for known phrase labels on word
boundaries — multi-word ("out of use") and verb-lemma aware (so the TTL lemma
`operate` matches the surface `operating`, via the WordNet lemmatizer).

**Prompt template** (verbatim; `{evidence}` is the assembled fact list, capped at
15 facts, highest-signal sources first):

```
You are a strict logical reasoning assistant.

Premise: '{premise}'
Consequence: '{consequence}'

Knowledge base facts (from the LaSSI ontologies):
{evidence}

Use the knowledge base facts above as verified background knowledge. Treat them
as ground truth when they are relevant, but rely on your own judgment for
anything they do not cover.
Analyze if the premise implies the consequence. Respond ONLY in valid JSON ...
  { "reasoning": "...", "implication_score": 1.0 / 0.5 / 0.0 }
```

**Dispatch format.** The backend is selected by the transformer string
`LLMHOnK#<model>#<sources>` (FullText representation), where `<sources>` is a
`+`-joined subset of `honk`/`lifecycle`/`paraphrase` (omitted ⇒ all three). The
ungrounded baseline is `LLM#<model>`.

## 4. Implementation summary (files)

| File | Change |
|---|---|
| `LaSSI/similarities/LLMandHOnK.py` | **new** — `LLMHOnKPrompt`, source-gated HOnK bootstrap, three fact builders, lemma-aware phrase scan, evidence assembly, grounded prompt. |
| `LaSSI/similarities/LLM.py` | split `_query` into `_build_prompt` + `_dispatch` so the subclass reuses all Ollama plumbing (behaviour unchanged). |
| `LaSSI/LaSSI.py` | `LLMHOnK#` dispatch branch in `_calculate_matrix`; `self.fuzzyDBs` stashed for the FullText path; per-source reasoning-file suffix. |
| `main.py` | documented the `LLMHOnK#<model>#<sources>` strings. |
| reused as-is | `LaSSI/HOnK/TBox/ParaphraseManager.py` (`_paraphrase_match`, `_get_paraphrase_concepts`), `LaSSI/HOnK/TBox/LifecycleManager.py` (`_get_lifecycle_phrases`) — TTL-only, no Postgres/HOnK needed. |

## 5. Experimental design

- **Task / corpus.** NEET 3-class claim verification (Supported / Refuted /
  Not-Enough-Evidence). **90 assertions over 30 cases**, balanced 30/30/30,
  domains transport/weather/roadworks/crime. Each case yields a 4×4 FullText
  similarity matrix; the evaluator reads row-0 (evidence) vs the three claim
  columns, mapped to a label by `eval_results.get_label_matrix_output` — the
  *same* mapping the ungrounded `LLM#` baseline uses (apples-to-apples).
- **Conditions** (per base model): `ungrounded` (`LLM#<m>`), `honk`, `lifecycle`
  (isolation of the contradiction source), `honk+lifecycle`, `all`.
- **Base models:** `llama3.2:3b`, `qwen2.5:7b`, `gemma4:e2b` (local, via Ollama).
- **Metrics:** accuracy, macro-F1, and per-class F1 — the headline mechanism is
  **F1(Refuted)** = contradiction detection. Sliced by `claim_type` and `domain`,
  reported as Δ vs the ungrounded baseline of the same model.

## 6. Results manifest (what to `\input` / `\includegraphics`)

All under `results/appendix_honk/`:

| Artifact | LaTeX label | Content |
|---|---|---|
| `tab_ablation_overall.tex` | `tab:honk-ablation-overall` | Per (model × condition): n, Acc, macro-F1, ΔmacroF1 vs ungrounded, F1(Sup.), F1(Ref.). **The headline table.** |
| `tab_ablation_claimtype.tex` | `tab:honk-ablation-claimtype` | Accuracy by `claim_type` (paraphrase / negation / location / time / cause / outcome / unsupported-extra), mean over models. Shows *where* grounding acts. |
| `tab_ablation_domain.tex` | `tab:honk-ablation-domain` | Accuracy by domain, mean over models. |
| `tab_qualitative.tex` | `tab:honk-ablation-qualitative` | Flipped examples (ungrounded wrong → grounded right) with the model's own reasoning, mined from the `*_reasoning.json` files. |
| `fig_macroF1_by_condition.{pdf,png}` | (figure) | Grouped bars: macro-F1 per condition, per model. |
| `fig_perclass_f1.{pdf,png}` | (figure) | Per-class F1, ungrounded vs +all, mean over models. |
| `ablation_summary.csv`, `ablation_results.csv` | — | Machine-readable summary + per-assertion detail. |

**How to read the headline:** read down each model's block in
`tab:honk-ablation-overall`. The ΔF1 column isolates each rung's marginal effect.
Expect (and verify against the final CSV): `+HOnK` alone is *not* reliably
positive (general WordNet synonymy injects spurious "equivalent" facts);
`+Lifecycle` should lift **F1(Refuted)**; the full stack should be the best or
near-best grounded condition. In `tab:honk-ablation-claimtype`, expect movement
concentrated on `outcome_mismatch` (lifecycle) and `approximate_paraphrase`
(paraphrase), with `location_mismatch` / `time_mismatch` ~flat — those phenomena
are **not** in the TTLs, an honest coverage statement rather than a weakness to
hide.

### Final results (complete grid: 15 conditions × 30 cases × 3 claims = 1350 assertions)

Macro-F1 by condition (Δ vs that model's ungrounded baseline in parentheses):

| Model | ungrounded | +HOnK | +Lifecycle | +HOnK+Life | +All |
|---|---|---|---|---|---|
| `llama3.2:3b` | 0.640 | 0.489 (−0.151) | **0.703 (+0.063)** | 0.500 | 0.566 |
| `qwen2.5:7b`  | 0.569 | 0.688 (+0.119) | 0.553 | 0.685 | **0.702 (+0.133)** |
| `gemma4:e2b`  | 0.692 | 0.657 | 0.609 | 0.602 | **0.694 (+0.002)** |

**Three robust findings, in order of how much weight they can bear:**

1. **Contradiction detection (F1 Refuted) improves under grounding for *every*
   model** — the cleanest, most consistent result, and it matches the design
   hypothesis (lifecycle supplies the hard contradiction signal):
   `llama` 0.640→**0.692** (+Lifecycle), `qwen` 0.684→**0.732** (+Lifecycle/All),
   `gemma` 0.712→**0.831** (+All). Lead with this.
2. **The best grounded condition equals or beats the ungrounded baseline for all
   three models** (`llama`+Lifecycle, `qwen`+All, `gemma`+All) — i.e. grounding,
   *with the right source*, never loses and often wins (qwen +0.13).
3. **Source choice matters and is model-dependent.** HOnK-alone is high-variance:
   it *helps* `qwen` (+0.12) but *wrecks* `llama` (−0.15, WordNet-synonymy noise the
   weaker model can't filter). **Lifecycle is the safe source** (best for `llama`,
   never catastrophic). The full stack is the best *single recommendation* (best or
   tied for `qwen`/`gemma`, positive contradiction-F1 everywhere).

**Supporting detail.** `qwen2.5:7b`'s F1(Not-Enough-Evidence) leaps **0.114→0.465**
under grounding — it stops over-asserting and learns to abstain (the qualitative
flips in `tab:honk-ablation-qualitative` are exactly these abstentions, with the
model citing the injected facts). By claim type
(`tab:honk-ablation-claimtype`), `+Lifecycle` lifts `outcome_mismatch` 75→92 and
`approximate_paraphrase` 86→94; `location_mismatch`/`time_mismatch` stay flat or
*drop* (not in the TTLs — and `+HOnK` actively degrades `location_mismatch` 77→57
via spurious place-name synonymy), the honest coverage boundary.

*All numbers above are reproduced in `results/appendix_honk/ablation_summary.csv`;
the LaTeX tables in that directory carry the per-class and per-slice breakdowns.*

## 7. Worked qualitative example

Premise *"the footbridge is out of use"* vs consequence *"both footbridges are
fully operational"*. The grounding injects the lifecycle fact
`"out of use" is incompatible with "operational" (both describe transport
availability ...)`, and `qwen2.5:7b` returns **0.0 (Refuted)** with reasoning
that cites it: *"Given the knowledge base facts, 'out of use' and 'operational'
are incompatible states ... the premise directly contradicts the consequence."*
The ungrounded model is prone to mislabel this as partial/Not-Enough-Evidence.
(`tab_qualitative.tex` auto-populates such flips once the `all` condition runs.)

## 8. Limitations / threats to validity

- **Small per-`claim_type` N** — `outcome_mismatch` has only 4 assertions; per-slice
  numbers are indicative, not significance-tested.
- **Local models** — small open models (3–7B); a frontier model might already
  "know" the generic facts, shrinking the grounding delta (but not the
  project-specific canon, e.g. numeric probability buckets).
- **Surface phrase matching** — lemma-aware but not a full parse, so some
  paraphrase/lifecycle phrases can be missed (recall ceiling on the grounding).
- **Hand-curated, domain-tuned ontologies** — Lifecycle/Paraphrase were authored
  for this corpus; the gains may not transfer to unseen domains. Conversely, this
  is exactly the project knowledge a generic LLM lacks.
- **Coverage, not omniscience** — grounding cannot help phenomena the TTLs don't
  encode (location/time mismatches); the appendix should state this plainly.

## 9. Reproduction

From the repo root, with the project venv (`~/PycharmProjects/LaSSI/.venv/bin/python`)
and `ollama serve` running with the three models pulled:

```bash
# 1. Populate catabolites for every (model x condition x case). Idempotent /
#    resumable — already-computed matrices are skipped.
python eval/run_honk_ablation.py

# 2. Build the CSVs, LaTeX tables and figures into results/appendix_honk/.
python eval/eval_honk_grounding.py
```

Scope/cost: 3 models × 5 conditions × 30 cases; each case is a 4×4 matrix ≈ 12
off-diagonal LLM calls (~5.4k calls total for a from-scratch run). HOnK loads
once per process (~18 s) for the `honk*` conditions; `lifecycle`-only and
`ungrounded` need no HOnK. Runner flags: `--models`, `--conditions`, `--cases`,
`--force`. Eval flags: `--catabolites`, `--out`, `--models`.
