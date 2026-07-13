"""Runner for the HOnK-grounding ablation (paper appendix).

Drives the FullText similarity pipeline over the NEET corpus for every
(base model x grounding condition), populating
``catabolites/<case>/confusion_matrices_FullText_<transformer>.json`` so that
``eval/eval_honk_grounding.py`` can build the appendix tables/figures.

Conditions (per base model):
  ungrounded        LLM#<m>                              (no grounding baseline)
  conceptnet        LLMHOnK#<m>#conceptnet               (ConceptNet-only ontology; run in its OWN invocation)
  honk              LLMHOnK#<m>#honk                     (general ontology only)
  lifecycle         LLMHOnK#<m>#lifecycle                (contradiction source, isolated)
  honk+lifecycle    LLMHOnK#<m>#honk+lifecycle
  all               LLMHOnK#<m>#honk+lifecycle+paraphrase

Notes
-----
* One long-lived process so HOnK loads once (the LLMHOnK backend bootstraps the
  HOnKSingleton lazily and it persists across cases).
* FullText needs no Postgres/NER, so ``disable_fuzzy_honk=True`` is passed; the
  backend loads HOnK from the local TTL/RocksDB cache (onStorage=False).
* Idempotent / resumable: a condition+case whose matrix file already exists is
  skipped unless ``--force`` (so an in-flight run is reused).

Run from the repo root with the project venv:
    ~/PycharmProjects/LaSSI/.venv/bin/python eval/run_honk_ablation.py
"""
import argparse
import os
import sys
import time

# This script lives in eval/, so a bare `python eval/run_honk_ablation.py` puts
# eval/ (not the repo root) on sys.path[0] and `import LaSSI` would resolve to
# the stale `pip install .` snapshot in site-packages (which lacks the LLMHOnK#
# dispatch branch). Force the repo root first so the live source is used.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pandas as pd

from LaSSI.Configuration import SentenceRepresentation
from LaSSI.LaSSI import LaSSI

DEFAULT_MODELS = ["llama3.2:3b", "qwen2.5:7b", "gemma4:e2b"]

# condition name -> transformer-string builder given a base model `m`
# NB: "conceptnet" grounds against the ConceptNet-only ontology build; the
# HOnK singleton binds one ontology per process, so run it in a SEPARATE
# invocation (e.g. `--conditions conceptnet`) from the honk-based conditions.
CONDITIONS = {
    "ungrounded":     lambda m: f"LLM#{m}",
    "conceptnet":     lambda m: f"LLMHOnK#{m}#conceptnet",
    "honk":           lambda m: f"LLMHOnK#{m}#honk",
    "lifecycle":      lambda m: f"LLMHOnK#{m}#lifecycle",
    "honk+lifecycle": lambda m: f"LLMHOnK#{m}#honk+lifecycle",
    "all":            lambda m: f"LLMHOnK#{m}#honk+lifecycle+paraphrase",
}

NEET_CSV = "neet/neet_v1.csv"
EVIDENCE_DIR = os.path.join("neet", "evidence_cases")
CATABOLITES = "catabolites"


def case_list_from_csv(csv_path=NEET_CSV):
    """Unique case stems referenced by the eval CSV (e.g. 'crime_001')."""
    df = pd.read_csv(csv_path)
    stems = df["item_id"].astype(str).str.rsplit("_", n=1).str[0]
    return sorted(stems.unique())


def matrix_path(case, transformer):
    """Where ex_post_explain writes the FullText matrix for this transformer.

    Mirrors LaSSI.ex_post_explain: experiment_name =
    'FullText_' + HuggingFace.split('/')[-1]. Our transformer strings contain no
    '/', so the whole string (incl. the '#<sources>' tag) is kept verbatim.
    """
    suffix = transformer.split("/")[-1]
    return os.path.join(CATABOLITES, case, "matrices",
                        f"confusion_matrices_FullText_{suffix}.json")


def run_one(case, transformer, conn, force):
    ds = os.path.join(EVIDENCE_DIR, f"{case}.yaml")
    if not os.path.exists(ds):
        print(f"  [skip] dataset missing: {ds}", flush=True)
        return "missing"
    out = matrix_path(case, transformer)
    if os.path.exists(out) and not force:
        print(f"  [have] {out}", flush=True)
        return "cached"
    t0 = time.time()
    pipeline = LaSSI(
        ds, conn, SentenceRepresentation.FullText,
        transformer=transformer,
        disable_fuzzy_honk=True,        # FullText needs no Postgres/NER
        use_multiprocessing=False,
        generate_png_matrix=False,
        force=force,
    )
    pipeline.run()
    pipeline.close()
    print(f"  [done] {out}  ({time.time() - t0:.1f}s)", flush=True)
    return "ran"


def main():
    ap = argparse.ArgumentParser(description="HOnK-grounding ablation runner")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS),
                    choices=list(CONDITIONS))
    ap.add_argument("--cases", nargs="+", default=None,
                    help="case stems (default: all from neet_v1.csv)")
    ap.add_argument("--connection", default="connection.yaml")
    ap.add_argument("--force", action="store_true",
                    help="recompute even if a matrix file already exists")
    args = ap.parse_args()

    cases = args.cases or case_list_from_csv()
    print(f"Models:     {args.models}")
    print(f"Conditions: {args.conditions}")
    print(f"Cases:      {len(cases)} ({cases[0]} .. {cases[-1]})")
    print(f"Total runs: {len(args.models) * len(args.conditions) * len(cases)}\n")

    counts = {"ran": 0, "cached": 0, "missing": 0}
    manifest = []
    for model in args.models:
        for cond in args.conditions:
            transformer = CONDITIONS[cond](model)
            print(f"=== {model} | {cond} | {transformer} ===", flush=True)
            for case in cases:
                status = run_one(case, transformer, args.connection, args.force)
                counts[status] = counts.get(status, 0) + 1
                if status in ("ran", "cached"):
                    manifest.append(matrix_path(case, transformer))

    print(f"\nDONE  ran={counts['ran']} cached={counts['cached']} "
          f"missing={counts['missing']}  matrices on disk={len(manifest)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
