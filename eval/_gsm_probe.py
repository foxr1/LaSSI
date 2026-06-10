"""Standalone GSM grammar probe: run a (possibly scratch) gsm_query over a
cached per-case gsmDB.txt WITHOUT the pipeline, and dump the result graphs.

Usage:
    python eval/_gsm_probe.py <case> [query_file] [out_dir]

Defaults: query_file = LaSSI/resources/gsm_query.txt,
out_dir = /tmp/gsm_probe/<case>. Prints a node/edge summary per graph;
the full result graphs land in <out_dir>/<i>/result.json.
"""
import json
import os
import sys

sys.path.insert(0, os.getcwd())


def summarise(graph):
    by_id = {n['id']: n for n in graph}
    lines = []
    for n in sorted(graph, key=lambda x: x['id']):
        props = n.get('properties', {})
        keyprops = {k: v for k, v in props.items()
                    if k in ('kernel', 'root', 'subjpass', 'actioned') or
                    k.replace('.', '').isdigit()}
        lines.append(f"  [{n['id']}] xi={n.get('xi')} ell={n.get('ell')} {keyprops}")
        for e in n.get('phi', []):
            child = e.get('score', {}).get('child')
            cxi = (by_id.get(child) or {}).get('xi')
            lines.append(f"        --{e.get('containment')}--> {child} {cxi}")
    return "\n".join(lines)


def main():
    case = sys.argv[1]
    query = sys.argv[2] if len(sys.argv) > 2 else "LaSSI/resources/gsm_query.txt"
    out = sys.argv[3] if len(sys.argv) > 3 else f"/tmp/gsm_probe/{case}"
    gsm_db = f"catabolites/{case}/gsmDB.txt"
    assert os.path.exists(gsm_db), gsm_db
    os.makedirs(out, exist_ok=True)

    n_graphs = open(gsm_db).read().count("~~") + 1

    from PyDatagramDB import DatagramDB
    d = DatagramDB(gsm_db, query, out,
                   full_server_output=False,
                   isSerializationFull=True,
                   opt_data_schema="pos\nSizeTAtt\nbegin\nSizeTAtt\nend\nSizeTAtt")
    d.run()

    for i in range(n_graphs):
        rf = os.path.join(out, str(i), "result.json")
        if not os.path.exists(rf):
            print(f"=== graph {i}: NO RESULT ===")
            continue
        with open(rf) as f:
            g = json.load(f)
        print(f"=== graph {i} ({len(g)} nodes) ===")
        print(summarise(g))


if __name__ == "__main__":
    main()
