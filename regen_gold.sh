#!/bin/zsh
# Regenerate NEET cases so the similarity suite can run.
# One LaSSI instance per process with use_multiprocessing=False: the
# multiprocessing+jpype(JVM) interaction deadlocks startJVM, and multiple
# LaSSI instances in one process can't restart the JVM — so we fork a fresh
# Python per case. Deleting internals.json invalidates meuDB/gsmDB via
# LaSSI._invalidate_stale_cache, so each case does a full pipeline pass.
#
# Usage:
#   zsh regen_gold.sh              # the 20 gold cases (similarities_neet.json)
#   zsh regen_gold.sh all          # all 30 evidence cases
#   zsh regen_gold.sh crime_004 …  # explicit case list
#   ~/PycharmProjects/LaSSI/.venv/bin/python -m unittest LaSSI/tests/test_similarities.py
cd /Users/fox/Documents/Ubuntu/LaSSI
PY=~/PycharmProjects/LaSSI/.venv/bin/python
CASES_GOLD=(crime_001 crime_002 crime_003 crime_004 crime_005 crime_006 crime_007 \
       transport_001 transport_002 transport_004 transport_005 transport_006 \
       weather_001 weather_002 weather_003 weather_004 weather_005 weather_006 weather_007 weather_008)
CASES_ALL=(crime_001 crime_002 crime_003 crime_004 crime_005 crime_006 crime_007 \
       roadworks_001 roadworks_002 roadworks_003 roadworks_004 roadworks_005 roadworks_006 roadworks_007 \
       transport_001 transport_002 transport_003 transport_004 transport_005 transport_006 transport_007 transport_008 \
       weather_001 weather_002 weather_003 weather_004 weather_005 weather_006 weather_007 weather_008)
if [ $# -eq 0 ]; then
  CASES=($CASES_GOLD)
elif [ "$1" = "all" ]; then
  CASES=($CASES_ALL)
else
  CASES=($@)
fi
cat > /tmp/_regen_one.py <<'PYEOF'
import os, sys
# The repo root MUST shadow the stale site-packages LaSSI snapshot: a script
# run from /tmp puts /tmp (not the cwd) on sys.path[0], so without this
# insert the regen silently runs old pipeline code.
sys.path.insert(0, os.getcwd())
import LaSSI as _l
assert os.path.dirname(os.path.dirname(os.path.abspath(_l.__file__))) == os.getcwd(), \
    f"LaSSI imported from {_l.__file__}, not the repo — aborting"
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
from LaSSI.LaSSI import LaSSI
from LaSSI.Configuration import SentenceRepresentation
c = sys.argv[1]
p = LaSSI(f"neet/evidence_cases/{c}.yaml", "connection.yaml",
          SentenceRepresentation.Logical, use_multiprocessing=False, generate_png_matrix=False)
p.run(); p.close()
PYEOF
for c in $CASES; do
  for f in logical_rewriting.json internals.json internals-bin.json string_rep.txt; do
    rm -f "catabolites/$c/$f"
  done
  rm -f "catabolites/$c/matrices/confusion_matrices_Logical.json" \
        "catabolites/$c/matrices/confusion_matrices_Logical.png" \
        "catabolites/$c/matrices/confusion_matrices_Logical_paper.png"
  rm -rf "catabolites/$c/SentenceRepresentation.Logical"
  echo "=== REGEN $c ($(date +%H:%M:%S)) ==="
  timeout 900 $PY /tmp/_regen_one.py "$c" > "/tmp/regen_$c.log" 2>&1
  [ -f "catabolites/$c/matrices/confusion_matrices_Logical.json" ] && echo "OK $c" || echo "MISSING $c (see /tmp/regen_$c.log)"
done
echo "=== REGEN DONE ==="
