import os
from LaSSI.structures.extended_fol.Formulae import FVariable, FNot, FAnd, FOr, FUnaryPredicate, FBinaryPredicate

_LIFECYCLE_NS = "https://ofox.co.uk/lifecycle#"
_LIFECYCLE_TTL_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "LifecycleStates.ttl")
)

_lifecycle_phrases_cache = None     # phrase_label_lower -> (dimension, partition)
_DESCRIPTIVE_PARTITION = "descriptive"

def _load_lifecycle_ttl() -> dict:
    """Parse LifecycleStates.ttl with pyoxigraph.

    Returns: {phrase_label_lower: (dimension_label, partition_label)}.
    Two phrases contradict iff they share a dimension and have differing partitions.
    """
    import pyoxigraph
    if not os.path.exists(_LIFECYCLE_TTL_PATH):
        return {}
    store = pyoxigraph.Store()
    store.load(path=_LIFECYCLE_TTL_PATH, format=pyoxigraph.RdfFormat.TURTLE)

    rdf_type   = pyoxigraph.NamedNode("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
    rdfs_label = pyoxigraph.NamedNode("http://www.w3.org/2000/01/rdf-schema#label")
    member_of  = pyoxigraph.NamedNode(_LIFECYCLE_NS + "memberOf")
    partition  = pyoxigraph.NamedNode(_LIFECYCLE_NS + "partition")
    dim_cls    = pyoxigraph.NamedNode(_LIFECYCLE_NS + "LifecycleDimension")
    phrase_cls = pyoxigraph.NamedNode(_LIFECYCLE_NS + "LifecyclePhrase")

    dim_labels: dict = {}
    for q in store.quads_for_pattern(None, rdf_type, dim_cls, None):
        for lq in store.quads_for_pattern(q.subject, rdfs_label, None, None):
            dim_labels[q.subject.value] = lq.object.value

    result: dict = {}
    for q in store.quads_for_pattern(None, rdf_type, phrase_cls, None):
        phrase_uri = q.subject
        label = dim_uri = part = None
        for lq in store.quads_for_pattern(phrase_uri, rdfs_label, None, None):
            label = lq.object.value
        for mq in store.quads_for_pattern(phrase_uri, member_of, None, None):
            dim_uri = mq.object.value
        for pq in store.quads_for_pattern(phrase_uri, partition, None, None):
            part = pq.object.value
        if label and dim_uri and part and dim_uri in dim_labels:
            result[label.strip().lower()] = (dim_labels[dim_uri], part)
    return result

def _get_lifecycle_phrases() -> dict:
    global _lifecycle_phrases_cache
    if _lifecycle_phrases_cache is None:
        _lifecycle_phrases_cache = _load_lifecycle_ttl()
    return _lifecycle_phrases_cache

def _extract_indicators_from_value(v, out: set, negated: bool = False) -> None:
    """Recursively extract `(phrase, assertive)` status candidates."""
    if isinstance(v, str):
        s = v.strip().lower()
        if s and not s.startswith("or("):
            out.add((s, not negated))
    elif isinstance(v, FVariable):
        if v.name:
            out.add((v.name.strip().lower(), not negated))
        if isinstance(v.specification, str):
            out.add((v.specification.strip().lower(), not negated))
        if v.cop is not None:
            _extract_indicators_from_value(v.cop, out, negated)
        if v.properties:
            for _, vs in v.properties:
                if isinstance(vs, tuple):
                    for inner in vs:
                        _extract_indicators_from_value(inner, out, negated)
                else:
                    _extract_indicators_from_value(vs, out, negated)
    elif isinstance(v, FNot):
        _extract_indicators_from_value(v.arg, out, not negated)
    elif isinstance(v, (FAnd, FOr)):
        for arg in getattr(v, "args", ()) or ():
            _extract_indicators_from_value(arg, out, negated)
    elif isinstance(v, (FBinaryPredicate, FUnaryPredicate)):
        out.update(_collect_status_indicators(v, negated=negated))

def _collect_status_indicators(formula, negated: bool = False) -> set:
    """Gather `(phrase, assertive)` pairs from a predicate."""
    indicators: set = set()
    if isinstance(formula, FNot):
        return _collect_status_indicators(formula.arg, negated=not negated)
    if not isinstance(formula, (FBinaryPredicate, FUnaryPredicate)):
        return indicators
    if formula.rel:
        indicators.add((formula.rel.strip().lower(), not negated))
    if formula.properties:
        for _, vs in formula.properties:
            if isinstance(vs, tuple):
                for v in vs:
                    _extract_indicators_from_value(v, indicators, negated)
            else:
                _extract_indicators_from_value(vs, indicators, negated)
    if isinstance(formula, FBinaryPredicate):
        if formula.src is not None:
            _extract_indicators_from_value(formula.src, indicators, negated)
        if formula.dst is not None:
            _extract_indicators_from_value(formula.dst, indicators, negated)
    elif isinstance(formula, FUnaryPredicate):
        if formula.arg is not None:
            _extract_indicators_from_value(formula.arg, indicators, negated)
    return indicators

def _lifecycle_partition_verdict(lhs, rhs):
    """Compare lhs and rhs through the lifecycle-states taxonomy."""
    phrases = _get_lifecycle_phrases()
    if not phrases:
        return None

    _dim_partitions: dict = {}
    for (dim, part) in phrases.values():
        if part == _DESCRIPTIVE_PARTITION:
            continue
        _dim_partitions.setdefault(dim, set()).add(part)

    def _invert_partition(dim, part):
        siblings = _dim_partitions.get(dim, set()) - {part}
        if len(siblings) == 1:
            return next(iter(siblings))
        return None

    def _phrase_marks(formula):
        out = set()
        def _add(dim_part, assertive):
            dim, part = dim_part
            if assertive:
                out.add((dim, part))
                return
            other = _invert_partition(dim, part)
            if other is not None:
                out.add((dim, other))

        for (ind, assertive) in _collect_status_indicators(formula):
            if ind in phrases:
                _add(phrases[ind], assertive)
                continue
            for tok in ind.split():
                if tok in phrases:
                    _add(phrases[tok], assertive)
        return out

    lhs_marks = _phrase_marks(lhs)
    rhs_marks = _phrase_marks(rhs)
    if not lhs_marks or not rhs_marks:
        return None

    def _by_dim(marks):
        out = {}
        for (dim, part) in marks:
            out.setdefault(dim, set()).add(part)
        return out

    lhs_dims = _by_dim(lhs_marks)
    rhs_dims = _by_dim(rhs_marks)
    asymmetric = False
    for dim in set(lhs_dims) | set(rhs_dims):
        lhs_parts = lhs_dims.get(dim, set())
        rhs_parts = rhs_dims.get(dim, set())
        if not lhs_parts or not rhs_parts:
            continue
        lhs_assertive = {p for p in lhs_parts if p != _DESCRIPTIVE_PARTITION}
        rhs_assertive = {p for p in rhs_parts if p != _DESCRIPTIVE_PARTITION}
        if lhs_assertive and rhs_assertive:
            if not (lhs_assertive & rhs_assertive):
                return 'contradiction'
        elif lhs_assertive or rhs_assertive:
            asymmetric = True
    return 'asymmetric' if asymmetric else None

def _lifecycle_partition_contradiction(lhs, rhs) -> bool:
    """Back-compat wrapper: True only for genuine partition contradictions."""
    return _lifecycle_partition_verdict(lhs, rhs) == 'contradiction'
