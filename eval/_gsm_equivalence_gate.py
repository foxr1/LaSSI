"""Engine-adoption gate: run the live grammar over every cached gsmDB.txt with
the CURRENTLY INSTALLED pydatagramdb and structurally compare each result graph
against the cached catabolites/<case>/datagramdb_output.json.

Usage:
    python eval/_gsm_equivalence_gate.py [case ...]    # default: all 30

Structural comparison = per-node (id, xi, ell, sorted properties, sorted
(containment, child) edges); JSON key/array order churn is ignored.
Exit 0 = all equivalent; 1 = any mismatch (listed).
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.getcwd())


def norm(graph):
    return sorted(
        (n['id'], tuple(n.get('xi') or []), tuple(n.get('ell') or []),
         tuple(sorted((str(k), str(v)) for k, v in (n.get('properties') or {}).items())),
         tuple(sorted((e.get('containment'), e.get('score', {}).get('child'))
                      for e in n.get('phi', []))))
        for n in graph)


def main():
    cases = sys.argv[1:] or sorted(
        os.path.basename(os.path.dirname(p))
        for p in glob.glob('catabolites/*/gsmDB.txt'))
    from PyDatagramDB import DatagramDB
    bad = []
    for case in cases:
        gsm_db = f'catabolites/{case}/gsmDB.txt'
        cached_path = f'catabolites/{case}/datagramdb_output.json'
        if not (os.path.exists(gsm_db) and os.path.exists(cached_path)):
            print(f"{case}: SKIP (missing inputs)")
            continue
        out = f'/tmp/gsm_gate/{case}'
        os.makedirs(out, exist_ok=True)
        d = DatagramDB(gsm_db, 'LaSSI/resources/gsm_query.txt', out,
                       full_server_output=False, isSerializationFull=True,
                       opt_data_schema="pos\nSizeTAtt\nbegin\nSizeTAtt\nend\nSizeTAtt")
        d.run()
        cached = json.load(open(cached_path))
        case_ok = True
        for i, cg in enumerate(cached):
            rf = os.path.join(out, str(i), 'result.json')
            if not os.path.exists(rf):
                bad.append(f"{case} g{i}: no result"); case_ok = False; continue
            if norm(cg) != norm(json.load(open(rf))):
                bad.append(f"{case} g{i}: STRUCTURAL DIFF"); case_ok = False
        print(f"{case}: {'OK' if case_ok else 'MISMATCH'} ({len(cached)} graphs)")
    print(f"--- {'ALL EQUIVALENT' if not bad else 'MISMATCHES:'} ---")
    for b in bad:
        print(" ", b)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
