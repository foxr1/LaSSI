__author__ = "Oliver R. Fox"
__copyright__ = "Copyright 2026, Oliver R. Fox"
__license__ = "GPL"
__version__ = "2.0"
__maintainer__ = "Oliver R. Fox"

"""Shared, dependency-free reader for ``raw_data/logical_analysis.json``.

`SentenceStructure.load_logical_analysis` returns only ``(types, rules)`` and
sits under the HOnK package; the classification sets below are needed by the
comparison engine (`ModelSearch`), the TBox reasoners (`SpatialReasoner`) and
the structural rewrites, so they live here where every layer can import them
without cycles. All sets are DERIVED from the JSON — to change a construct's
behaviour, edit `logical_analysis.json`, never a Python set.
"""

import json
import os
from functools import lru_cache

_JSON_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "raw_data", "logical_analysis.json"))


@lru_cache(maxsize=1)
def load_logical_analysis_json():
    with open(_JSON_PATH) as fh:
        return json.load(fh)


@lru_cache(maxsize=1)
def kernel_context_keys():
    """Property keys whose construct has ``attachTo: Kernel`` in
    ``logical_analysis.json`` (the logical-context types), minus the declared
    ``similarity_semantics.kernel_context_excluded`` constructs, expanded with
    ``key_spelling_aliases``. Upper-cased to match eFOL property keys.

    These are the keys ModelSearch's implication guard treats as event-level
    logical context: when one predicate carries one of these and the other
    doesn't, the predicates describe semantically distinct events."""
    data = load_logical_analysis_json()
    types = data.get("types", {})
    sem = data.get("similarity_semantics", {})
    excluded = {str(k).upper() for k in sem.get("kernel_context_excluded", [])}
    keys = {
        name.upper() for name, specs in types.items()
        if any(s.get("attachTo") == "Kernel" for s in specs)
    } - excluded
    for canonical, aliases in sem.get("key_spelling_aliases", {}).items():
        if str(canonical).upper() in keys:
            keys |= {str(a).upper() for a in aliases}
    return frozenset(keys)


@lru_cache(maxsize=1)
def kernel_context_monotonicity():
    """Split of `kernel_context_keys()` by the per-spec ``monotonicity``
    declaration in ``logical_analysis.json`` (see `_doc_monotonicity` there):

    - ``restrictive`` (default): intersective event modifier — more-specific
      (with the key) entails less-specific (without). The guard blocks
      LHS=>RHS when RHS asserts the key and LHS lacks it.
    - ``intensional``: non-veridical operator (MODALITY) — the modal does NOT
      entail the factual. The guard blocks LHS=>RHS when LHS asserts the key
      and RHS lacks it.

    Returns ``(restrictive_keys, intensional_keys)`` partitioning the
    kernel-context set; spelling aliases follow their canonical key."""
    data = load_logical_analysis_json()
    types = data.get("types", {})
    sem = data.get("similarity_semantics", {})
    keys = kernel_context_keys()
    intensional = set()
    for name, specs in types.items():
        if name.upper() not in keys:
            continue
        if any(s.get("monotonicity") == "intensional" for s in specs):
            intensional.add(name.upper())
    for canonical, aliases in sem.get("key_spelling_aliases", {}).items():
        if str(canonical).upper() in intensional:
            intensional |= {str(a).upper() for a in aliases}
    return frozenset(keys - intensional), frozenset(intensional)


@lru_cache(maxsize=1)
def paraphrastic_slots():
    """``similarity_semantics.paraphrastic_slots``: kernel property key ->
    the canonical slot it paraphrases. The guard equates the SLOTS only; the
    values are still compared point-wise."""
    sem = load_logical_analysis_json().get("similarity_semantics", {})
    return {
        str(k).upper(): str(v).upper()
        for k, v in sem.get("paraphrastic_slots", {}).items()
    }


@lru_cache(maxsize=None)
def spatial_relation_labels(subset="all"):
    """Spatial-relation type labels from the ``spatial_relations`` block.

    ``movement``  -> displacement-or-stay relations;
    ``relation``  -> movement + proximity (name-compatibility set);
    ``all``       -> movement + proximity + path (every spatial label)."""
    block = load_logical_analysis_json().get("spatial_relations", {})
    movement = tuple(block.get("movement", ()))
    proximity = tuple(block.get("proximity", ()))
    path = tuple(block.get("path", ()))
    if subset == "movement":
        return frozenset(movement)
    if subset == "relation":
        return frozenset(movement + proximity)
    if subset == "all":
        return frozenset(movement + proximity + path)
    raise ValueError(f"Unknown spatial_relations subset {subset!r}")
