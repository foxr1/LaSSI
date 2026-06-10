"""Comparison-only matrix recompute: loop cases in ONE process (HOnK singleton
loads once). Only valid when internals.json/logical_rewriting.json are intact —
delete SentenceRepresentation.Logical + matrices/confusion_matrices_Logical.json
per case first."""
import os, sys
sys.path.insert(0, os.getcwd())
import LaSSI as _l
assert os.path.dirname(os.path.dirname(os.path.abspath(_l.__file__))) == os.getcwd(), \
    f"LaSSI imported from {_l.__file__}, not the repo - aborting"
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
from LaSSI.LaSSI import LaSSI
from LaSSI.Configuration import SentenceRepresentation

for c in sys.argv[1:]:
    print(f"=== RECOMPUTE {c} ===", flush=True)
    p = LaSSI(f"neet/evidence_cases/{c}.yaml", "connection.yaml",
              SentenceRepresentation.Logical, use_multiprocessing=False,
              generate_png_matrix=False)
    p.run()
    p.close()
print("=== RECOMPUTE DONE ===")
