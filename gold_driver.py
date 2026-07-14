"""One-process gold regression driver. Regenerates the 17 gold matrices
(HOnK loads once), clearing kernel/eFOL/comparison caches per case but keeping
the slow meuDB/gsm/grammar caches. Run from repo root with the venv python."""
import os
import shutil
import sys

from LaSSI.Configuration import SentenceRepresentation
from LaSSI.LaSSI import LaSSI

GOLD = [
    "crime_001", "crime_002", "crime_003", "crime_005", "crime_006", "crime_007",
    "transport_001", "transport_005", "transport_006",
    "weather_001", "weather_002", "weather_003", "weather_004", "weather_005",
    "weather_006", "weather_007", "weather_008",
]


def clear_case(case):
    base = os.path.join("catabolites", case)
    for f in ("internals.json", "logical_rewriting.json", "string_rep.txt"):
        p = os.path.join(base, f)
        if os.path.exists(p):
            os.remove(p)
    for matrices_file in (
        os.path.join(base, "matrices", "confusion_matrices_Logical.json"),
        os.path.join(base, "confusion_matrices_Logical.json"),  # backward compat
    ):
        if os.path.exists(matrices_file):
            os.remove(matrices_file)
    d = os.path.join(base, "SentenceRepresentation.Logical")
    if os.path.isdir(d):
        shutil.rmtree(d)


if __name__ == '__main__':
    cases = sys.argv[1:] or GOLD
    for case in cases:
        clear_case(case)
        ds = os.path.join("neet", "evidence_cases", f"{case}.yaml")
        print(f"==== RUN {case} ====", flush=True)
        pipeline = LaSSI(ds, "connection.yaml", SentenceRepresentation.Logical,
                         useId=True, use_multiprocessing=True, transformer='all-MiniLM-L6-v2')
        pipeline.run()
        pipeline.close()
    print("==== ALL CASES DONE ====", flush=True)
