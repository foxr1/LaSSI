"""Diff the current per-case Logical similarity matrices against a snapshot.

Usage:
    python eval/diff_matrices.py catabolites-baseline/P0 [case ...]

Snapshot layout: <snapshot_dir>/<case>.json (one confusion_matrices_Logical.json
per case, as written by the Phase-0 snapshot loop). Current matrices are read
from catabolites/<case>/matrices/confusion_matrices_Logical.json.

For every cell where the value changed, prints:
    case | i->j | old | new | gold | sentence_i => sentence_j
where gold is the value in LaSSI/tests/assertions/similarities_neet.json
("null" = any value strictly in (0,1); "-" = case not gold-tracked).
Exit code 1 if any cell changed, 0 otherwise.
"""
import json
import os
import sys

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLD_PATH = os.path.join(REPO, "LaSSI", "tests", "assertions", "similarities_neet.json")
CASES_DIR = os.path.join(REPO, "neet", "evidence_cases")
CATABOLITES = os.path.join(REPO, "catabolites")


def load_sentences(case):
    path = os.path.join(CASES_DIR, f"{case}.yaml")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        data = yaml.safe_load(f)
    if isinstance(data, dict):
        for key in ("sentences", "all"):
            if key in data and isinstance(data[key], list):
                return [str(s) for s in data[key]]
        return [str(v) for v in data.values()]
    if isinstance(data, list):
        return [str(s) for s in data]
    return []


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    snapshot_dir = sys.argv[1]
    only_cases = set(sys.argv[2:])

    gold = json.load(open(GOLD_PATH)) if os.path.exists(GOLD_PATH) else {}

    changed = 0
    compared = 0
    missing = []
    for snap_file in sorted(os.listdir(snapshot_dir)):
        if not snap_file.endswith(".json"):
            continue
        case = snap_file[:-len(".json")]
        if only_cases and case not in only_cases:
            continue
        cur_path = os.path.join(CATABOLITES, case, "matrices", "confusion_matrices_Logical.json")
        if not os.path.exists(cur_path):
            missing.append(case)
            continue
        old_m = json.load(open(os.path.join(snapshot_dir, snap_file)))
        new_m = json.load(open(cur_path))
        compared += 1
        sentences = load_sentences(case)
        gold_m = gold.get(case)
        if len(old_m) != len(new_m):
            print(f"{case}: MATRIX SHAPE CHANGED {len(old_m)}x? -> {len(new_m)}x?")
            changed += 1
            continue
        for i in range(len(old_m)):
            for j in range(len(old_m[i])):
                old_v, new_v = old_m[i][j], new_m[i][j]
                if isinstance(old_v, float) and isinstance(new_v, float) and abs(old_v - new_v) < 1e-9:
                    continue
                if old_v == new_v:
                    continue
                gold_v = gold_m[i][j] if gold_m else "-"
                si = sentences[i] if i < len(sentences) else f"S{i}"
                sj = sentences[j] if j < len(sentences) else f"S{j}"
                print(f"{case} | {i}->{j} | old={old_v} new={new_v} gold={gold_v}\n"
                      f"    {si!r} => {sj!r}")
                changed += 1
    if missing:
        print(f"NO CURRENT MATRIX for: {', '.join(missing)}")
    print(f"--- {compared} cases compared, {changed} changed cells"
          f"{', ' + str(len(missing)) + ' missing' if missing else ''} ---")
    return 1 if (changed or missing) else 0


if __name__ == "__main__":
    sys.exit(main())
