"""One-case instrumented regen for the generalisation audit.

Runs a single gold case through the full pipeline (clearing the kernel/eFOL/
comparison caches so structural rewrites re-run), while recording which
structural-rewrite rules fire. Writes the fired/seen counts to
catabolites/<case>/_rule_firing.json so a parent can union them across the
corpus for dead-rule detection. Regenerates confusion_matrices_Logical.json so
the similarity suite can re-verify behaviour-neutrality.

Forked one-per-case (JVM can't be restarted within a single process).
Run from the repo root with the venv python:  python _audit_trace_one.py <case>
"""
import collections
import json
import os
import shutil
import sys

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from LaSSI.Configuration import SentenceRepresentation
from LaSSI.LaSSI import LaSSI
from LaSSI.ner.structural_rewrites.base import RuleRegistry

FIRED = collections.Counter()
SEEN = collections.Counter()

_orig_apply_phase = RuleRegistry.apply_phase


def _traced_apply_phase(self, kernel, phase, ctx):
    # Mirror the non-trace path of RuleRegistry.apply_phase, recording whether
    # each rule's (side-effect-free) matches() fires. Equivalent output.
    for rule in self.rules_for_phase(phase):
        SEEN[rule.name] += 1
        bindings = rule.matches(kernel, ctx)
        if bindings is None:
            continue
        FIRED[rule.name] += 1
        kernel = rule.apply(kernel, bindings, ctx)
    return kernel


RuleRegistry.apply_phase = _traced_apply_phase


def clear_case(case):
    base = os.path.join("catabolites", case)
    for f in ("internals.json", "internals-bin.json", "logical_rewriting.json",
              "string_rep.txt", "confusion_matrices_Logical.json",
              "confusion_matrices_Logical.png"):
        p = os.path.join(base, f)
        if os.path.exists(p):
            os.remove(p)
    d = os.path.join(base, "SentenceRepresentation.Logical")
    if os.path.isdir(d):
        shutil.rmtree(d)


if __name__ == "__main__":
    case = sys.argv[1]
    clear_case(case)
    ds = os.path.join("neet", "evidence_cases", f"{case}.yaml")
    pipeline = LaSSI(ds, "connection.yaml", SentenceRepresentation.Logical,
                     useId=True, use_multiprocessing=False, transformer="all-MiniLM-L6-v2",
                     generate_png_matrix=False)
    pipeline.run()
    pipeline.close()
    out = os.path.join("catabolites", case, "_rule_firing.json")
    with open(out, "w") as fh:
        json.dump({"case": case, "fired": dict(FIRED), "seen": dict(SEEN)}, fh, indent=2)
    print(f"==== DONE {case}: {len(FIRED)} rules fired ====", flush=True)
