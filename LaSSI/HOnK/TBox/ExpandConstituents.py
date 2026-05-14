import json
import json
import os.path
import pickle
import re
import sys

from LaSSI.HOnK.HOnK import CasusHappening, HOnKSingleton
from LaSSI.structures.extended_fol.Enums import PairwiseCases
from LaSSI.structures.extended_fol.Formulae import FVariable, FNot, FBinaryPredicate, FUnaryPredicate
from LaSSI.structures.extended_fol.ModelSearch import ModelSearch, ModelSearchBasis


def isImplication(x):
    return x == CasusHappening.GENERAL_IMPLICATION or x == CasusHappening.LOSE_SPEC_IMPLICATION or x == CasusHappening.INSTANTIATION_IMPLICATION or x == CasusHappening.MISSING_1ST_IMPLICATION


d_transformCaseWhenOneArgIsNegated = None


def transformCaseWhenOneArgIsNegated(orig: CasusHappening):
    """Negation of a more-valued logic (where we have more than just satisfiability or not)"""
    global d_transformCaseWhenOneArgIsNegated
    if d_transformCaseWhenOneArgIsNegated is None:
        d_transformCaseWhenOneArgIsNegated = {CasusHappening.NONE: CasusHappening.NONE,
                                              CasusHappening.INDIFFERENT: CasusHappening.INDIFFERENT,
                                              CasusHappening.EQUIVALENT: CasusHappening.EXCLUSIVES,
                                              CasusHappening.EXCLUSIVES: CasusHappening.EQUIVALENT,
                                              CasusHappening.GENERAL_IMPLICATION: CasusHappening.INDIFFERENT,
                                              CasusHappening.LOSE_SPEC_IMPLICATION: CasusHappening.INDIFFERENT,
                                              CasusHappening.INSTANTIATION_IMPLICATION: CasusHappening.INDIFFERENT,
                                              CasusHappening.MISSING_1ST_IMPLICATION: CasusHappening.INDIFFERENT}
    return d_transformCaseWhenOneArgIsNegated[orig]

def isExistential(x):
    if x is None:
        return False
    return isinstance(x, FVariable) and x.name[0] == "?" and x.name[1:].isdigit()


_PARA_NS = "https://ofox.co.uk/paraphrase#"
_PARA_TTL_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "Paraphrase.ttl")
)

_paraphrase_concepts_cache = None   # loaded lazily; None = not yet loaded
_asked_pairs: set = set()           # session-dedup for interactive prompt

_LIFECYCLE_NS = "https://ofox.co.uk/lifecycle#"
_LIFECYCLE_TTL_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "LifecycleStates.ttl")
)

_lifecycle_phrases_cache = None     # phrase_label_lower -> (dimension, partition)


def _load_paraphrase_ttl() -> dict:
    """Parse Paraphrase.ttl with pyoxigraph and return a PARAPHRASE_CONCEPTS dict."""
    import pyoxigraph
    if not os.path.exists(_PARA_TTL_PATH):
        return {}
    store = pyoxigraph.Store()
    store.load(path=_PARA_TTL_PATH, format=pyoxigraph.RdfFormat.TURTLE)

    rdf_type   = pyoxigraph.NamedNode("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
    rdfs_label = pyoxigraph.NamedNode("http://www.w3.org/2000/01/rdf-schema#label")
    member_of  = pyoxigraph.NamedNode(_PARA_NS + "memberOf")
    form_pred  = pyoxigraph.NamedNode(_PARA_NS + "form")
    concept_cls = pyoxigraph.NamedNode(_PARA_NS + "ParaphraseConcept")
    phrase_cls  = pyoxigraph.NamedNode(_PARA_NS + "ParaphrasePhrase")

    concept_labels: dict = {}
    for q in store.quads_for_pattern(None, rdf_type, concept_cls, None):
        uri = q.subject
        for lq in store.quads_for_pattern(uri, rdfs_label, None, None):
            concept_labels[uri.value] = lq.object.value

    result: dict = {}
    for q in store.quads_for_pattern(None, rdf_type, phrase_cls, None):
        phrase_uri = q.subject
        label = concept_uri_str = form = None
        for lq in store.quads_for_pattern(phrase_uri, rdfs_label, None, None):
            label = lq.object.value
        for mq in store.quads_for_pattern(phrase_uri, member_of, None, None):
            concept_uri_str = mq.object.value
        for fq in store.quads_for_pattern(phrase_uri, form_pred, None, None):
            form = fq.object.value
        if label and concept_uri_str and form and concept_uri_str in concept_labels:
            concept_name = concept_labels[concept_uri_str]
            result.setdefault(concept_name, set()).add((form, label.strip().lower()))
    return result


def _get_paraphrase_concepts() -> dict:
    global _paraphrase_concepts_cache
    if _paraphrase_concepts_cache is None:
        _paraphrase_concepts_cache = _load_paraphrase_ttl()
    return _paraphrase_concepts_cache


def _reload_paraphrase_concepts():
    global _paraphrase_concepts_cache
    _paraphrase_concepts_cache = None


def _paraphrase_concept_of(v):
    concepts = _get_paraphrase_concepts()
    if isinstance(v, FVariable) and v.name:
        nm = v.name.strip().lower()
        for concept, members in concepts.items():
            if ("name", nm) in members:
                return concept
    elif isinstance(v, FNot):
        inner = v.arg
        if isinstance(inner, FVariable) and inner.name:
            nm = inner.name.strip().lower()
            for concept, members in concepts.items():
                if ("fnot_name", nm) in members:
                    return concept
    return None


def _paraphrase_match(lhs, rhs):
    cl = _paraphrase_concept_of(lhs)
    if cl is None:
        return False
    return cl == _paraphrase_concept_of(rhs)


def _extract_var_info(v):
    """Return (default_form, name_lower, display_str) for a variable-like expr."""
    if isinstance(v, FVariable) and v.name and not isExistential(v):
        nm = v.name.strip().lower()
        return ("name", nm, f'"{nm}"')
    if isinstance(v, FNot) and isinstance(v.arg, FVariable):
        inner = v.arg
        if inner.name and not isExistential(inner):
            nm = inner.name.strip().lower()
            return ("fnot_name", nm, f'NOT("{nm}")')
    return (None, None, None)


def _slugify(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]", "_", s.strip().lower()).strip("_")


def _to_camel(s: str) -> str:
    return "".join(w.capitalize() for w in re.split(r"[\s_]+", s.strip()))


def _append_phrase_to_ttl(concept_uri_local: str, form: str, name: str):
    slug = _slugify(name)
    lines = [
        f"\npara:phrase_{slug}\n",
        f"    a para:ParaphrasePhrase ;\n",
        f"    rdfs:label {json.dumps(name)} ;\n",
        f"    para:memberOf para:{concept_uri_local} ;\n",
        f"    para:form {json.dumps(form)} .\n",
    ]
    with open(_PARA_TTL_PATH, "a") as fh:
        fh.writelines(lines)


def _interactive_paraphrase_prompt(lhs, rhs):
    """
    If running interactively and the pair is novel, ask the user whether lhs
    and rhs should be recorded as paraphrases in Paraphrase.ttl.
    Returns True if the user confirmed and the TTL was updated.
    """
    if not sys.stdin.isatty():
        return False

    lhs_form, lhs_name, lhs_disp = _extract_var_info(lhs)
    rhs_form, rhs_name, rhs_disp = _extract_var_info(rhs)
    if lhs_name is None or rhs_name is None:
        return False

    pair_key = (lhs_disp, rhs_disp)
    if pair_key in _asked_pairs:
        return False
    _asked_pairs.add(pair_key)

    print(f"\n[Paraphrase?] {lhs_disp} vs {rhs_disp} → INDIFFERENT")
    ans = input("  Are these paraphrases of each other? [y/N]: ").strip().lower()
    if ans != "y":
        return False

    # Show existing concepts so the user can reuse one
    concepts = _get_paraphrase_concepts()
    if concepts:
        print(f"  Existing concepts: {', '.join(sorted(concepts))}")
    concept_name = input("  Concept name (existing or new): ").strip().lower()
    if not concept_name:
        print("  (skipped — no concept name given)")
        return False

    concept_uri_local = _to_camel(concept_name)
    new_concept = concept_name not in concepts

    if new_concept:
        example = input("  Example sentence (press Enter to skip): ").strip()
        lines = [
            f"\npara:{concept_uri_local}\n",
            f"    a para:ParaphraseConcept ;\n",
            f"    rdfs:label {json.dumps(concept_name)} ;\n",
            f"    rdfs:comment {json.dumps(example)} .\n",
        ]
        with open(_PARA_TTL_PATH, "a") as fh:
            fh.writelines(lines)

    # Ask/confirm the form for each side (defaulting to the natural Python type)
    lhs_form_in = input(
        f"  Form for {lhs_disp} [name/fnot_name, default: {lhs_form}]: "
    ).strip().lower()
    if lhs_form_in in ("name", "fnot_name"):
        lhs_form = lhs_form_in

    rhs_form_in = input(
        f"  Form for {rhs_disp} [name/fnot_name, default: {rhs_form}]: "
    ).strip().lower()
    if rhs_form_in in ("name", "fnot_name"):
        rhs_form = rhs_form_in

    _append_phrase_to_ttl(concept_uri_local, lhs_form, lhs_name)
    _append_phrase_to_ttl(concept_uri_local, rhs_form, rhs_name)
    _reload_paraphrase_concepts()
    print(f"  → Saved to Paraphrase.ttl under concept '{concept_name}'.")
    return True


_GEO_TYPES = {"LOC", "GPE", "FAC"}


_SPATIAL_MOVEMENT_LABELS = frozenset({"stay in place", "motion to place", "motion from place"})


def _strip_spatial_type_properties(props):
    """Remove `type` entries whose values are spatial-movement labels.

    These labels (`stay in place`, `motion to place`, ...) are emitted from
    the verb's preposition signature (`at` vs `to` vs `from`), not from the
    geo entity itself. For ex-post comparison of geo-typed FVariables the
    preposition is a surface artifact: "to Haymarket Station" and
    "Haymarket Station" denote the same place, just relative to different
    verbs' aspectual perspectives. Stripping them here lets equivalent geo
    names compare EQUIVALENT in the property-level pass even when one side
    surfaced a preposition and the other didn't. "near place" is preserved
    because downstream logic uses it as a genuine semantic distinction
    (proximity, not co-location)."""
    if not props:
        return props
    new_props = []
    for key, val in props:
        if key == "type":
            vals = val if isinstance(val, tuple) else (val,)
            filtered = tuple(
                v for v in vals
                if not (isinstance(v, str) and v.strip().lower() in _SPATIAL_MOVEMENT_LABELS)
            )
            if not filtered:
                continue
            if len(filtered) == 1:
                new_props.append((key, filtered[0]))
            else:
                new_props.append((key, filtered))
        else:
            new_props.append((key, val))
    return frozenset(new_props)


def _canonicalize_geo_fvar(fvar):
    """Fold a facility-noun `specification` into `name` for geo-typed FVariables.

    Stanza NER inconsistently parses things like "Monkseaton station": sometimes
    as a single FAC entity ("Monkseaton station"), sometimes as a LOC
    "Monkseaton" with "station" attached as the `extra`/specification. Both
    refer to the same place, so for ex-post comparison we normalise them to the
    folded form ("Monkseaton station", no specification) before comparing.
    The original FVariable is left untouched downstream of this comparison.

    Also strips preposition-derived spatial type properties (`stay in place`,
    `motion to place`) so geo entities that name-match aren't blocked from
    EQUIVALENT by surface preposition differences (e.g. `Haymarket Station`
    vs `to Haymarket Metro station`).
    """
    if not isinstance(fvar, FVariable):
        return fvar
    if (fvar.type or "").upper() not in _GEO_TYPES and (fvar.type or "").upper() != "SPACE":
        return fvar
    new_props = _strip_spatial_type_properties(fvar.properties)
    spec = fvar.specification
    if not isinstance(spec, str) or not spec or not fvar.name:
        if new_props != fvar.properties:
            return FVariable(fvar.name, fvar.type, fvar.specification, fvar.cop, fvar.id, new_props, fvar.spec_negation, fvar.meta, fvar.asAll)
        return fvar
    spec_first_word = spec.strip().split()[0].lower() if spec.strip() else ""
    if not spec_first_word:
        if new_props != fvar.properties:
            return FVariable(fvar.name, fvar.type, fvar.specification, fvar.cop, fvar.id, new_props, fvar.spec_negation, fvar.meta, fvar.asAll)
        return fvar
    facility_nouns = HOnKSingleton.get().getFacilityNouns()
    if spec_first_word not in facility_nouns:
        if new_props != fvar.properties:
            return FVariable(fvar.name, fvar.type, fvar.specification, fvar.cop, fvar.id, new_props, fvar.spec_negation, fvar.meta, fvar.asAll)
        return fvar
    if spec_first_word in fvar.name.lower().split():
        # Already folded, avoid duplication.
        return FVariable(fvar.name, fvar.type, None, fvar.cop, fvar.id, new_props, fvar.spec_negation, fvar.meta, fvar.asAll)
    return FVariable(f"{fvar.name} {spec.strip()}", fvar.type, None, fvar.cop, fvar.id, new_props, fvar.spec_negation, fvar.meta, fvar.asAll)


def _strip_geo_generic_suffix(name: str | None) -> str | None:
    """Return `name` with a trailing GeoSuffixNoun word removed, or `name` unchanged.

    The suffix vocabulary is sourced from the GeoSuffixNoun class in HOnK so
    that the list is maintained in one place (raw_data/nouns/geo_suffix_nouns.txt)
    rather than hardcoded here.
    """
    if name is None:
        return None
    words = name.strip().split()
    if len(words) > 1 and words[-1].lower() in HOnKSingleton.get().getGeoSuffixNouns():
        return " ".join(words[:-1])
    return name


def _has_near_place(fvar) -> bool:
    """True iff the FVariable's type property includes 'near place', either
    directly or as a disjunct of an FOr object."""
    from LaSSI.structures.extended_fol.Formulae import FOr as _FOr
    if not isinstance(fvar, FVariable) or not fvar.properties:
        return False
    for key, val in fvar.properties:
        if key != "type":
            continue
        vals = val if isinstance(val, tuple) else (val,)
        for v in vals:
            if isinstance(v, _FOr):
                for arg in getattr(v, "args", ()) or ():
                    arg_name = getattr(arg, "name", None)
                    if arg_name and "near place" in arg_name.lower():
                        return True
                continue
            v_str = v if isinstance(v, str) else getattr(v, "name", None)
            if v_str and "near place" in v_str.lower():
                return True
    return False


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


def _extract_indicators_from_value(v, out: set) -> None:
    """Recursively extract lowercase status candidates from a property value."""
    if isinstance(v, str):
        s = v.strip().lower()
        if s and not s.startswith("or("):
            out.add(s)
    elif isinstance(v, FVariable):
        if v.name:
            out.add(v.name.strip().lower())
        if isinstance(v.specification, str):
            out.add(v.specification.strip().lower())
        if v.properties:
            for _, vs in v.properties:
                if isinstance(vs, tuple):
                    for inner in vs:
                        _extract_indicators_from_value(inner, out)
                else:
                    _extract_indicators_from_value(vs, out)


def _collect_status_indicators(formula) -> set:
    """Gather candidate status-bearing terms (lowercase) from a predicate's
    rel + property values (including nested FVariable name/specification/type)."""
    indicators: set = set()
    if not isinstance(formula, (FBinaryPredicate, FUnaryPredicate)):
        return indicators
    if formula.rel:
        indicators.add(formula.rel.strip().lower())
    if formula.properties:
        for _, vs in formula.properties:
            if isinstance(vs, tuple):
                for v in vs:
                    _extract_indicators_from_value(v, indicators)
            else:
                _extract_indicators_from_value(vs, indicators)
    return indicators


_DESCRIPTIVE_PARTITION = "descriptive"


def _lifecycle_partition_verdict(lhs, rhs):
    """Compare lhs and rhs through the lifecycle-states taxonomy.

    Returns one of:
      'contradiction' — both sides assert state on the same dimension with
                        no shared non-descriptive partition (e.g.
                        'under_investigation' [open] vs 'conclude' [closed]).
      'asymmetric'    — exactly one side asserts a non-descriptive partition
                        on a dimension and the other side is silent there or
                        only carries a descriptive (event-facet) marker
                        (e.g. 'conclude' [closed] vs 'involve' [descriptive]).
      None            — no lifecycle info, or both sides agree on partition.

    The descriptive partition is reserved for verbs that talk about the event
    without committing to a lifecycle stage (involve, describe, mention).
    They count as lifecycle-aware indicators so an open/closed claim on the
    other side surfaces as an asymmetry, but they never contradict open or
    closed directly.
    """
    phrases = _get_lifecycle_phrases()
    if not phrases:
        return None
    lhs_marks = {phrases[t] for t in _collect_status_indicators(lhs) if t in phrases}
    rhs_marks = {phrases[t] for t in _collect_status_indicators(rhs) if t in phrases}
    # Both sides must register at least one lifecycle indicator before any
    # verdict can be emitted.  A side with zero marks is "silent" rather than
    # "contradicting" — important inside constituent-expansion comparisons,
    # where sub-atoms derived from a lifecycle-bearing predicate often drop
    # the verb entirely (e.g. `bicycle theft happened` without `conclude`).
    # Treating those silent sub-atoms as if they disagreed with their parent
    # would make every sentence carrying a lifecycle verb self-contradict
    # through its own expansion.
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
            # Only one side has any indicator on this dimension — silence vs
            # presence is not, on its own, a contradiction (see comment above).
            continue
        lhs_assertive = {p for p in lhs_parts if p != _DESCRIPTIVE_PARTITION}
        rhs_assertive = {p for p in rhs_parts if p != _DESCRIPTIVE_PARTITION}
        if lhs_assertive and rhs_assertive:
            if not (lhs_assertive & rhs_assertive):
                return 'contradiction'
        elif lhs_assertive or rhs_assertive:
            # One side asserts open/closed, the other only carries a
            # descriptive marker on the same dimension — asymmetric.
            asymmetric = True
    return 'asymmetric' if asymmetric else None


def _lifecycle_partition_contradiction(lhs, rhs) -> bool:
    """Back-compat wrapper: True only for genuine partition contradictions."""
    return _lifecycle_partition_verdict(lhs, rhs) == 'contradiction'


def _space_near_place_names(formula):
    """Concrete near-place value names from a predicate's SPACE property.

    Picks up FVariable entries whose `type` property includes 'near place',
    either directly or as a disjunct of an FOr.  Existentials and unnamed
    entries are skipped — we want named micro-locations such as 'Parking Area'
    or 'Central Station area'.
    """
    from LaSSI.structures.extended_fol.Formulae import FOr as _FOr
    if not isinstance(formula, (FBinaryPredicate, FUnaryPredicate)):
        return None
    if not formula.properties:
        return None
    space_vals = None
    for k, vs in formula.properties:
        if k == "SPACE":
            space_vals = vs if isinstance(vs, tuple) else (vs,)
            break
    if not space_vals:
        return None
    names = set()
    for v in space_vals:
        if not isinstance(v, FVariable):
            continue
        if v.name is None or (isinstance(v.name, str) and v.name.startswith("?")):
            continue
        if not v.properties:
            continue
        type_vals = None
        for kk, vv in v.properties:
            if kk == "type":
                type_vals = vv if isinstance(vv, tuple) else (vv,)
                break
        if not type_vals:
            continue
        is_near = False
        for t in type_vals:
            if isinstance(t, _FOr):
                for arg in getattr(t, "args", ()) or ():
                    arg_name = getattr(arg, "name", None)
                    if arg_name and "near place" in arg_name.lower():
                        is_near = True
                        break
                if is_near:
                    break
                continue
            t_str = t if isinstance(t, str) else (t.name if isinstance(t, FVariable) else None)
            if t_str and "near place" in str(t_str).lower():
                is_near = True
                break
        if is_near:
            # Strip generic geo-suffix words (e.g. "area", "district") so that
            # "Haymarket area" and "Haymarket" are treated as the same location.
            stripped = _strip_geo_generic_suffix(v.name)
            names.add(stripped.strip().lower())
    return names if names else None


def _space_mismatch_contradiction(lhs, rhs) -> bool:
    """True iff both predicates have concrete near-place SPACE values that
    do not overlap by name or by HOnK name-equivalence.

    Captures the case where two records share a city (stay-in-place) but
    disagree on the specific micro-location — e.g. 'Parking Area' vs
    'Central Station area' both inside Newcastle.  Same-city overlap on the
    GPE/stay-in-place dimension is not enough on its own to justify partial
    similarity credit when the named near-place differs.
    """
    lhs_names = _space_near_place_names(lhs)
    rhs_names = _space_near_place_names(rhs)
    if not lhs_names or not rhs_names:
        return False
    if lhs_names & rhs_names:
        return False
    if not HOnKSingleton.isReady():
        return True
    kb = HOnKSingleton.get()
    for ln in lhs_names:
        for rn in rhs_names:
            eq = kb.name_eq(ln, rn)
            if eq == CasusHappening.EQUIVALENT or isImplication(eq):
                return False
    return True


def _is_change_of_state_relation(rel):
    if rel is None or not HOnKSingleton.isReady():
        return False
    try:
        change_verbs = HOnKSingleton.get().getChangeOfStateVerbs() or set()
    except Exception:
        return False
    return str(rel).lower() in {str(v).lower() for v in change_verbs}


def _compare_single_prop_val(d, lhs_val, rhs_val):
    """Compare one property value (str, FVariable, FOr) in the lhs→rhs direction."""
    from LaSSI.structures.extended_fol.Formulae import FOr, FVariable as _FVar
    if lhs_val == rhs_val:
        return CasusHappening.EQUIVALENT

    def _name_match(a, b):
        if a == b:
            return True
        a_name = a if isinstance(a, str) else (a.name if isinstance(a, _FVar) else None)
        b_name = b if isinstance(b, str) else (b.name if isinstance(b, _FVar) else None)
        if a_name is not None and b_name is not None and a_name == b_name:
            return True
        return False

    if isinstance(rhs_val, FOr):
        # lhs is a disjunct of rhs → lhs is the more specific claim → lhs implies rhs
        if any(_name_match(lhs_val, arg) for arg in rhs_val.args):
            return CasusHappening.GENERAL_IMPLICATION
    if isinstance(lhs_val, FOr):
        # rhs is a disjunct of lhs → lhs is more general → lhs does not imply rhs
        if any(_name_match(rhs_val, arg) for arg in lhs_val.args):
            return CasusHappening.INDIFFERENT

    if isinstance(lhs_val, _FVar) and isinstance(rhs_val, _FVar):
        return compare_variable(d, lhs_val, rhs_val)
    if isinstance(lhs_val, str) and isinstance(rhs_val, str):
        return HOnKSingleton.get().name_eq(lhs_val, rhs_val)
    if isinstance(lhs_val, str) and isinstance(rhs_val, _FVar):
        return HOnKSingleton.get().name_eq(lhs_val, rhs_val.name)
    if isinstance(lhs_val, _FVar) and isinstance(rhs_val, str):
        return HOnKSingleton.get().name_eq(lhs_val.name, rhs_val)
    return CasusHappening.INDIFFERENT


def _is_syntactic_fvar_property_key(key) -> bool:
    """True for property keys that are syntactic/positional artefacts of parsing.

    Dependency-position integers (e.g. 4, 5, 9) appear as keys in FVariable
    properties alongside their preposition/conjunction value (e.g. 9→'of',
    4→'or').  These carry no semantic content — the semantic role is already
    captured by the 'type' property ('near place', 'stay in place', etc.) and
    the 'extra' property.  Determiners ('det') and punctuation ('punct') are
    likewise purely syntactic.

    Filtering these keys prevents syntactic surface differences (e.g. the
    presence of '(9:of)' on a nested Newcastle FVariable, or '(det:the)' on
    a Haymarket_area entity) from breaking logical equivalence comparisons
    that should succeed on semantic grounds.
    """
    if isinstance(key, int):
        return True
    if isinstance(key, str) and key.isdigit():
        return True
    if key in {"det", "punct"}:
        return True
    return False


def _compare_fvar_properties(d, lhs_props, rhs_props):
    """Compare FVariable.properties frozensets with FOr-awareness.

    Returns the CasusHappening for the lhs→rhs direction.  A key present only
    in lhs means lhs is more constrained (more specific) → GENERAL_IMPLICATION.
    A key present only in rhs means rhs is more specific → INDIFFERENT for
    the lhs→rhs direction.

    Syntactic/positional keys (integer dependency-position labels, determiners,
    punctuation) are excluded from comparison — see _is_syntactic_fvar_property_key.
    """
    if lhs_props == rhs_props:
        return CasusHappening.EQUIVALENT
    lhs_dict = {k: v for k, v in dict(lhs_props).items() if not _is_syntactic_fvar_property_key(k)}
    rhs_dict = {k: v for k, v in dict(rhs_props).items() if not _is_syntactic_fvar_property_key(k)}
    all_keys = set(lhs_dict.keys()) | set(rhs_dict.keys())
    results = []
    for key in all_keys:
        if key in lhs_dict and key in rhs_dict:
            lhs_vals = lhs_dict[key] if isinstance(lhs_dict[key], tuple) else (lhs_dict[key],)
            rhs_vals = rhs_dict[key] if isinstance(rhs_dict[key], tuple) else (rhs_dict[key],)
            pair_results = [
                _compare_single_prop_val(d, lv, rv)
                for lv in lhs_vals for rv in rhs_vals
            ]
            results.append(simplifyConstituentsAcross(pair_results))
        elif key in lhs_dict:
            results.append(CasusHappening.GENERAL_IMPLICATION)
        else:
            results.append(CasusHappening.INDIFFERENT)
    return simplifyConstituentsAcross(results)


def compare_variable(d, lhs, rhs):
    cp = (lhs, rhs)
    if (cp not in d) and (lhs == rhs):
        d[cp] = CasusHappening.EQUIVALENT
    if cp in d:
        return d[cp]
    if (lhs == rhs):
        val = CasusHappening.EQUIVALENT
    elif lhs is None:
        val = CasusHappening.MISSING_1ST_IMPLICATION if not isExistential(rhs) else CasusHappening.EQUIVALENT
    elif rhs is None:
        val = CasusHappening.INDIFFERENT if not isExistential(lhs) else CasusHappening.EQUIVALENT
    elif (lhs == FNot(rhs)) or (rhs == FNot(lhs)):
        val = CasusHappening.EXCLUSIVES
    elif _paraphrase_match(lhs, rhs):
        val = CasusHappening.EQUIVALENT
    elif isinstance(lhs, FNot):
        val = transformCaseWhenOneArgIsNegated(compare_variable(d, lhs.arg, rhs))
    elif isinstance(rhs, FNot):
        val = transformCaseWhenOneArgIsNegated(compare_variable(d, lhs, rhs.arg))
    elif (not isinstance(lhs, FVariable)) or (not isinstance(rhs, FVariable)):
        return CasusHappening.INDIFFERENT
    else:
        assert isinstance(lhs, FVariable)
        assert isinstance(rhs, FVariable)
        # Normalise geo-typed FVariables so that `Monkseaton station` (a single
        # FAC entity from Stanza) and `Monkseaton[extra:station]` (a LOC with a
        # facility-noun specification) compare equivalent. The folded form is
        # used purely for this comparison; the underlying formulae are kept
        # intact so the specification semantics survive elsewhere.
        lhs = _canonicalize_geo_fvar(lhs)
        rhs = _canonicalize_geo_fvar(rhs)
        kb = HOnKSingleton.get()
        lhs_type = lhs.type or ""
        rhs_type = rhs.type or ""
        # Case-fold equivalence shortcut: when two FVariables share a name
        # modulo case, treat them as the same entity.  Previously gated on
        # neither side being a geo type, which mis-fires when the upstream
        # parser assigns asymmetric type tags to the same named entity (e.g.
        # S1's "Parking Area" comes through as type='verb' from Stanza while
        # S2's "parking area" is type='LOC').  Falling through to a case-
        # sensitive name_eq in that situation breaks otherwise-trivial
        # paraphrase implications.
        if (lhs.name is not None and rhs.name is not None
                and lhs.name != rhs.name and lhs.name.lower() == rhs.name.lower()):
            nameEQ = CasusHappening.EQUIVALENT
        else:
            nameEQ = kb.name_eq(lhs.name, rhs.name)
        # Fallback for geo names that differ only by a generic spatial suffix
        # (e.g. "Haymarket" vs "Haymarket area").  Only applies when at least one
        # side actually has the suffix so we never silently conflate unrelated names.
        if (nameEQ == CasusHappening.INDIFFERENT
                and lhs_type in _GEO_TYPES and rhs_type in _GEO_TYPES
                and lhs.name is not None and rhs.name is not None):
            lhs_stripped = _strip_geo_generic_suffix(lhs.name)
            rhs_stripped = _strip_geo_generic_suffix(rhs.name)
            if lhs_stripped != lhs.name or rhs_stripped != rhs.name:
                stripped_eq = kb.name_eq(lhs_stripped, rhs_stripped)
                if stripped_eq == CasusHappening.EQUIVALENT:
                    # Both sides encode "near place" → pragmatically equivalent.
                    # Otherwise the "area" form is genuinely broader → implication.
                    if _has_near_place(lhs) and _has_near_place(rhs):
                        nameEQ = CasusHappening.EQUIVALENT
                    else:
                        nameEQ = CasusHappening.GENERAL_IMPLICATION
        specEQ = kb.name_eq(lhs.specification, rhs.specification)
        specEQInv = kb.name_eq(rhs.specification, lhs.specification)
        if lhs.spec_negation != rhs.spec_negation:
            specEQ = transformCaseWhenOneArgIsNegated(specEQ)
        copCompareInv = compare_variable(d, rhs.cop, lhs.cop)
        val = CasusHappening.INDIFFERENT
        if (nameEQ == specEQ) and (specEQ == copCompareInv) and (lhs.asAll == rhs.asAll):
            if lhs.properties == rhs.properties:
                d[cp] = specEQ
                return d[cp]
            prop_cmp = _compare_fvar_properties(d, lhs.properties, rhs.properties)
            if prop_cmp == CasusHappening.EQUIVALENT:
                d[cp] = specEQ
                return d[cp]
            elif specEQ == CasusHappening.EQUIVALENT:
                d[cp] = prop_cmp
                return d[cp]
            else:
                result = simplifyConstituentsAcross({specEQ, prop_cmp})
                d[cp] = result
                return result
        if nameEQ == CasusHappening.INDIFFERENT:
            if lhs.asAll:
                if rhs.asAll:
                    nameAgainstSpec = kb.name_eq(lhs.name, rhs.specification)
                    if nameAgainstSpec == CasusHappening.EQUIVALENT and lhs.name is not None and rhs.specification is not None:
                        val = CasusHappening.INSTANTIATION_IMPLICATION
                else:
                    flipNameEq = kb.name_eq(rhs.name, lhs.name)
                    nameAgainstSpec = kb.name_eq(rhs.name, lhs.specification)
                    if nameAgainstSpec == CasusHappening.EQUIVALENT and rhs.name is not None and lhs.specification is not None:
                        val = CasusHappening.INSTANTIATION_IMPLICATION
                    elif isImplication(flipNameEq):
                        val = flipNameEq
            else:
                if not rhs.asAll:
                    nameAgainstSpec = kb.name_eq(rhs.name, lhs.specification)
                    if nameAgainstSpec == CasusHappening.EQUIVALENT and rhs.name is not None and lhs.specification is not None:
                        val = CasusHappening.INSTANTIATION_IMPLICATION
                else:
                    val = CasusHappening.INDIFFERENT
        elif nameEQ == CasusHappening.EQUIVALENT:
            if (specEQ == copCompareInv):
                val = specEQ if ((lhs.asAll == rhs.asAll) or (lhs.asAll)) and (not isImplication(specEQ)) else CasusHappening.INDIFFERENT
            elif (specEQ == CasusHappening.EQUIVALENT):
                if copCompareInv == CasusHappening.MISSING_1ST_IMPLICATION:
                    val = CasusHappening.LOSE_SPEC_IMPLICATION
                else:
                    val = copCompareInv
            else:
                if specEQ == CasusHappening.MISSING_1ST_IMPLICATION:
                    val = CasusHappening.INSTANTIATION_IMPLICATION if lhs.asAll else CasusHappening.INDIFFERENT
                else:
                    if rhs.asAll:
                        val = specEQ
                    else:
                        val = specEQInv if rhs.specification is None else specEQ
        elif isImplication(nameEQ):
            nameAgainstSpec = kb.name_eq(lhs.name, rhs.specification)
            if (specEQ == copCompareInv) and (specEQ == CasusHappening.EQUIVALENT):
                val = nameEQ if (not rhs.asAll) and lhs.asAll else CasusHappening.INDIFFERENT ## If everything is equivalent, then it is implying as the arguments are
            elif (specEQ == CasusHappening.EQUIVALENT) and (copCompareInv == CasusHappening.EXCLUSIVES):
                val = CasusHappening.EXCLUSIVES
            elif lhs.specification is None and nameAgainstSpec == CasusHappening.EQUIVALENT:
                val = CasusHappening.INSTANTIATION_IMPLICATION if lhs.asAll else CasusHappening.INDIFFERENT
        elif nameEQ == CasusHappening.EXCLUSIVES:
            if (specEQ == copCompareInv) and (specEQ == CasusHappening.EQUIVALENT):
                val = CasusHappening.EXCLUSIVES
    if val == CasusHappening.INDIFFERENT:
        _interactive_paraphrase_prompt(lhs, rhs)
        # Re-check after potential TTL update — user may have added the pair
        if _paraphrase_match(lhs, rhs):
            val = CasusHappening.EQUIVALENT
    d[cp] = val
    return d[cp]


def simplifyConstituentsAcross(constituentCollection):
    if isinstance(constituentCollection, CasusHappening):
        return constituentCollection
    if CasusHappening.INDIFFERENT in constituentCollection:
        return CasusHappening.INDIFFERENT
    elif CasusHappening.EXCLUSIVES in constituentCollection:
        return CasusHappening.EXCLUSIVES
    elif CasusHappening.MISSING_1ST_IMPLICATION in constituentCollection:
        if not CasusHappening.GENERAL_IMPLICATION in constituentCollection and \
                not CasusHappening.INSTANTIATION_IMPLICATION in constituentCollection and \
                not CasusHappening.LOSE_SPEC_IMPLICATION in constituentCollection:
            return CasusHappening.MISSING_1ST_IMPLICATION
        else:
            return CasusHappening.GENERAL_IMPLICATION
    elif CasusHappening.INSTANTIATION_IMPLICATION in constituentCollection:
        if not CasusHappening.GENERAL_IMPLICATION in constituentCollection and \
                not CasusHappening.MISSING_1ST_IMPLICATION in constituentCollection and \
                not CasusHappening.LOSE_SPEC_IMPLICATION in constituentCollection:
            return CasusHappening.INSTANTIATION_IMPLICATION
        else:
            return CasusHappening.GENERAL_IMPLICATION
    elif CasusHappening.LOSE_SPEC_IMPLICATION in constituentCollection:
        if not CasusHappening.GENERAL_IMPLICATION in constituentCollection and \
                not CasusHappening.MISSING_1ST_IMPLICATION in constituentCollection and \
                not CasusHappening.INSTANTIATION_IMPLICATION in constituentCollection:
            return CasusHappening.LOSE_SPEC_IMPLICATION
        else:
            return CasusHappening.GENERAL_IMPLICATION
    elif CasusHappening.GENERAL_IMPLICATION in constituentCollection:
        return CasusHappening.GENERAL_IMPLICATION
    elif CasusHappening.INDIFFERENT in constituentCollection:
        return CasusHappening.INDIFFERENT
    elif CasusHappening.EQUIVALENT in constituentCollection:
        return CasusHappening.EQUIVALENT
    else:
        return CasusHappening.INDIFFERENT


def simplifyConstituents(constituentCollection):
    """
    This function calculates the simplication of the constituent collection to the set of elements required to
    boil down a set of elements to one single constituent
    """
    if isinstance(constituentCollection, CasusHappening):
        return constituentCollection
    elif CasusHappening.EXCLUSIVES in constituentCollection:
        return CasusHappening.EXCLUSIVES
    elif CasusHappening.EQUIVALENT in constituentCollection:
        return CasusHappening.EQUIVALENT
    elif CasusHappening.MISSING_1ST_IMPLICATION in constituentCollection:
        if not CasusHappening.GENERAL_IMPLICATION in constituentCollection and \
                not CasusHappening.INSTANTIATION_IMPLICATION in constituentCollection and \
                not CasusHappening.LOSE_SPEC_IMPLICATION in constituentCollection:
            return CasusHappening.MISSING_1ST_IMPLICATION
        else:
            return CasusHappening.GENERAL_IMPLICATION
    elif CasusHappening.INSTANTIATION_IMPLICATION in constituentCollection:
        if not CasusHappening.GENERAL_IMPLICATION in constituentCollection and \
                not CasusHappening.MISSING_1ST_IMPLICATION in constituentCollection and \
                not CasusHappening.LOSE_SPEC_IMPLICATION in constituentCollection:
            return CasusHappening.INSTANTIATION_IMPLICATION
        else:
            return CasusHappening.GENERAL_IMPLICATION
    elif CasusHappening.LOSE_SPEC_IMPLICATION in constituentCollection:
        if not CasusHappening.GENERAL_IMPLICATION in constituentCollection and \
                not CasusHappening.MISSING_1ST_IMPLICATION in constituentCollection and \
                not CasusHappening.INSTANTIATION_IMPLICATION in constituentCollection:
            return CasusHappening.LOSE_SPEC_IMPLICATION
        else:
            return CasusHappening.GENERAL_IMPLICATION
    elif CasusHappening.GENERAL_IMPLICATION in constituentCollection:
        return CasusHappening.GENERAL_IMPLICATION
    else:
        return CasusHappening.INDIFFERENT


def is_direct_subset(kv1, kv2):
    d1 = []
    d2 = []
    for k, t1 in kv1:
        for x in t1:
            d1.append((k,x))
    for k, t2 in kv2:
        for x in t2:
            d2.append((k, x))
    return set(d1).issubset(set(d2))


def test_pairwise_sentence_similarity(d, x, y, store=True, shift=True):
    # if shift:
    #     if (y, x) in d:
    #         test_shift = d[(y, x)]
    #     else:
    #         test_shift = test_pairwise_sentence_similarity(d, y, x, store,  False)
    #     if test_shift == CasusHappening.EQUIVALENT or test_shift == CasusHappening.EXCLUSIVES:
    #         d[(x, y)] = test_shift
    #         return test_shift
    val = CasusHappening.NONE
    if y is None and x is None:
        val = CasusHappening.EQUIVALENT
    elif y is None:
        val = CasusHappening.GENERAL_IMPLICATION
    elif x is None:
        val = CasusHappening.INDIFFERENT
    elif (x == y):
        val = CasusHappening.EQUIVALENT
    elif (isinstance(x, FNot) and isinstance(y, FNot)):
        val = test_pairwise_sentence_similarity(d, x.arg, y.arg, False, False)
        if isImplication(val):
            val = CasusHappening.INDIFFERENT
    elif (x == FNot(y)) or (y == FNot(x)):
        val = CasusHappening.EXCLUSIVES
    elif isinstance(x, FNot):
        val = transformCaseWhenOneArgIsNegated(test_pairwise_sentence_similarity(d, x.arg, y, False, False))
    elif isinstance(y, FNot):
        val = transformCaseWhenOneArgIsNegated(test_pairwise_sentence_similarity(d, x, y.arg, False, False))
    else:
        if (x.meta != y.meta):
            val = CasusHappening.INDIFFERENT
        else:
            assert isinstance(x, FBinaryPredicate) or isinstance(x, FUnaryPredicate)
            assert isinstance(y, FBinaryPredicate) or isinstance(y, FUnaryPredicate)
            xprop = set() if x.properties is None else x.properties
            yprop = set() if y.properties is None else y.properties
            keyCmp, keyCmpInv = {}, {}
            keys = set(map(lambda z: z[0], xprop)).union(map(lambda z: z[0], yprop))
            hasDirectSubset = False
            dLHS = dict(xprop)
            dRHS = dict(yprop)
            if (is_direct_subset(xprop, yprop) and len(xprop)>0) or (len(yprop) == 0 and len(xprop) > 0):
                keyCmpElements = CasusHappening.GENERAL_IMPLICATION
                keyCmpElementsInv = CasusHappening.INDIFFERENT
                hasDirectSubset = True
            # elif set(dLHS.keys()).issubset(set(dRHS.keys())) and set(dLHS.keys()) != set(dRHS.keys()):
            #     # LHS keys are a strict subset of RHS keys: RHS carries
            #     # property obligations (e.g. CAUSATION) that LHS makes no
            #     # claim about.  The forward direction (x→y) cannot license
            #     # those extras → INDIFFERENT.  The inverse direction (y→x)
            #     # can drop the extras → LOSE_SPEC_IMPLICATION (rule 5).
            #     keyCmpElements = CasusHappening.INDIFFERENT
            #     keyCmpElementsInv = CasusHappening.LOSE_SPEC_IMPLICATION
            #     hasDirectSubset = True
            else:
            # if is_direct_subset(yprop, xprop):
            #     keyCmpElements = CasusHappening.GENERAL_IMPLICATION
            #     keyCmpElementsInv = CasusHappening.INDIFFERENT
            #     hasDirectSubset = True
            # else:
                # dLHS = dict(xprop)
                # dRHS = dict(yprop)
                def _value_in_other(val, other_dict):
                    for ok in other_dict:
                        for ov in other_dict[ok]:
                            comparison = compare_variable(d, val, ov)
                            reverse_comparison = compare_variable(d, ov, val)
                            if (
                                    comparison == CasusHappening.EQUIVALENT
                                    or isImplication(comparison)
                                    or isImplication(reverse_comparison)):
                                return True
                    return False

                for key in keys:
                    if key in dLHS and key in dRHS:
                        # For each lhs value find its best match across all rhs
                        # values (OR), then require every lhs value to have a
                        # match (AND via simplifyConstituentsAcross).  This
                        # prevents a spurious EQUIVALENT when two sets share one
                        # member but differ on others, e.g. [A,B] vs [A,C].
                        lhs_bests = [
                            simplifyConstituents([compare_variable(d, xx, yy) for yy in dRHS[key]])
                            for xx in dLHS[key]
                        ]
                        rhs_bests = [
                            simplifyConstituents([compare_variable(d, yy, xx) for xx in dLHS[key]])
                            for yy in dRHS[key]
                        ]
                        keyCmp[key] = simplifyConstituentsAcross(lhs_bests) if lhs_bests else CasusHappening.EQUIVALENT
                        keyCmpInv[key] = simplifyConstituentsAcross(rhs_bests) if rhs_bests else CasusHappening.EQUIVALENT
                    elif key in dLHS:
                        soft = any(_value_in_other(xx, dRHS) for xx in dLHS[key])
                        keyCmp[key] = CasusHappening.INSTANTIATION_IMPLICATION if soft else CasusHappening.INDIFFERENT
                        keyCmpInv[key] = CasusHappening.GENERAL_IMPLICATION
                    else:
                        soft = any(_value_in_other(yy, dLHS) for yy in dRHS[key])
                        keyCmp[key] = CasusHappening.GENERAL_IMPLICATION
                        keyCmpInv[key] = CasusHappening.INSTANTIATION_IMPLICATION if soft else CasusHappening.INDIFFERENT
                if len(keyCmp) > 0:
                    keyCmpElements = simplifyConstituentsAcross({keyCmp[key] for key in keyCmp})
                    keyCmpElementsInv = simplifyConstituentsAcross({keyCmpInv[key] for key in keyCmpInv})
                else:
                    keyCmpElements, keyCmpElementsInv = CasusHappening.EQUIVALENT, CasusHappening.EQUIVALENT
            antonymRelationContradiction = False
            # Lifecycle-state mismatch.  Two flavours, both bypass the regular
            # relation/property comparison and force EXCLUSIVES (rules 1+3):
            #   1. Direct contradiction — both sides assert an open/closed
            #      partition and they don't overlap (e.g. under_investigation
            #      vs conclude).
            #   2. Asymmetric assertion — one side asserts a partition while
            #      the other is silent or only carries a descriptive
            #      (event-facet) marker like 'involve'.  Silent-vs-asserted
            #      pairs over the same event must not be granted the 0.5
            #      independence credit a BDD would otherwise hand out.
            # Source: LifecycleStates.ttl partitions.
            _lifecycle_verdict = _lifecycle_partition_verdict(x, y)
            # SPACE near-place divergence (rule 4): two records sharing only
            # their city but disagreeing on the named micro-location should
            # not collect partial credit either.
            _space_mismatch = _space_mismatch_contradiction(x, y)

            _causation_mismatch = False
            if hasattr(x, 'properties') and hasattr(y, 'properties') and x.properties and y.properties:
                x_caus = [v for k, v in x.properties if k in ("CAUSATION", "CAUSE")]
                y_caus = [v for k, v in y.properties if k in ("CAUSATION", "CAUSE")]
                if x_caus and y_caus:
                    # If both have a cause, but evaluating them yields INDIFFERENT, it's a hard contradiction
                    caus_cmp = simplifyConstituents([compare_variable(d, xc, yc) for xc in x_caus for yc in y_caus])
                    if caus_cmp == CasusHappening.INDIFFERENT:
                        _causation_mismatch = True

            if _lifecycle_verdict == 'contradiction' or _space_mismatch or _causation_mismatch:
                val = CasusHappening.EXCLUSIVES
                antonymRelationContradiction = True
            elif isinstance(x, FBinaryPredicate) and isinstance(y, FBinaryPredicate):
                # Compare relation names through the ontology, not by string
                # equality — so antonyms like close/open trigger EXCLUSIVES
                # propagation when both arguments coincide.
                relCmp = HOnKSingleton.get().name_eq(x.rel, y.rel) if x.rel != y.rel else CasusHappening.EQUIVALENT
                if relCmp == CasusHappening.INDIFFERENT:
                    val = CasusHappening.INDIFFERENT
                else:
                    srcCmp = compare_variable(d, x.src, y.src)
                    dstCmp = compare_variable(d, x.dst, y.dst)
                    if relCmp == CasusHappening.EXCLUSIVES:
                        # For change-of-state predicates (`open`/`close`),
                        # the affected object is the semantic anchor.  A
                        # concrete cause/agent on one side and an existential
                        # source on the other should not mask the contradiction.
                        if dstCmp == CasusHappening.EQUIVALENT and (
                                srcCmp == CasusHappening.EQUIVALENT
                                or _is_change_of_state_relation(x.rel)
                                or _is_change_of_state_relation(y.rel)):
                            val = CasusHappening.EXCLUSIVES
                            antonymRelationContradiction = True
                        else:
                            val = CasusHappening.INDIFFERENT
                    elif (srcCmp == CasusHappening.INDIFFERENT):
                        val = CasusHappening.INDIFFERENT
                    else:
                        if (dstCmp == CasusHappening.INDIFFERENT):
                            val = CasusHappening.INDIFFERENT
                        else:
                            keyComparisonOutcome = compare_variable(d, x.src, y.src)
                            copKeyComparisonOutcome = compare_variable(d, x.src.cop if hasattr(x.src, "cop") else None, y.src.cop if hasattr(y.src, "cop") else None)
                            if (srcCmp == CasusHappening.EXCLUSIVES) and (dstCmp == CasusHappening.EXCLUSIVES):
                                val = CasusHappening.INDIFFERENT
                            elif (srcCmp == CasusHappening.EXCLUSIVES) and (dstCmp != CasusHappening.INDIFFERENT):
                                val = CasusHappening.EXCLUSIVES
                            elif (dstCmp == CasusHappening.EXCLUSIVES) and (srcCmp != CasusHappening.INDIFFERENT):
                                val = CasusHappening.EXCLUSIVES
                            elif srcCmp == CasusHappening.EQUIVALENT:
                                val = dstCmp
                            elif dstCmp == CasusHappening.EQUIVALENT:
                                val = srcCmp
                            else:
                                val = simplifyConstituents({srcCmp, dstCmp})
            elif isinstance(y, FUnaryPredicate) and isinstance(x, FUnaryPredicate):
                relCmp = HOnKSingleton.get().name_eq(x.rel, y.rel) if x.rel != y.rel else CasusHappening.EQUIVALENT
                if relCmp == CasusHappening.INDIFFERENT:
                    val = CasusHappening.INDIFFERENT
                else:
                    argCmp = compare_variable(d, x.arg, y.arg)
                    if relCmp == CasusHappening.EXCLUSIVES:
                        if argCmp == CasusHappening.EQUIVALENT:
                            val = CasusHappening.EXCLUSIVES
                            antonymRelationContradiction = True
                        else:
                            val = CasusHappening.INDIFFERENT
                    else:
                        val = argCmp
                keyComparisonOutcome = compare_variable(d, x.arg, y.arg)
                copKeyComparisonOutcome = compare_variable(d, x.arg.cop if hasattr(x.arg, "cop") else None, y.arg.cop if hasattr(y.arg, "cop") else None)
            else:
                raise ValueError("Unexpected comparison between " + str(x) + " and" + str(y))
            if antonymRelationContradiction:
                # Antonym contradiction over equivalent arguments must not be
                # downgraded by secondary property mismatches.
                pass
            elif val != CasusHappening.INDIFFERENT:
                # Forward implication x ⇒ y requires every RHS obligation to
                # be matched by some LHS value (multi-valued properties such as
                # SPACE/TIME are conjunctive on the RHS event).  keyCmpInv
                # captures that direction: if any RHS value has no LHS match,
                # the forward implication cannot stand and must fall back to
                # INDIFFERENT — otherwise an existential lhs spec match
                # (e.g. Haymarket [of] Newcastle ↔ Newcastle) silently subsumes
                # an unmatched RHS sibling (Eldon Square).
                # An obligation is genuinely unmet only when the LHS also fails to
                # cover the RHS for that key (keyCmp[k] is not an implication).
                # If keyCmp[k] IS an implication, the LHS is more specific and
                # fully satisfies the RHS property — even if the inverse direction
                # is INDIFFERENT (RHS is more general, e.g. OR type vs near place).
                rhs_obligation_unmet = any(
                    v == CasusHappening.INDIFFERENT
                    and not isImplication(keyCmp.get(k, CasusHappening.INDIFFERENT))
                    for k, v in keyCmpInv.items()
                )
                if val == CasusHappening.EQUIVALENT:
                    if keyComparisonOutcome == CasusHappening.EQUIVALENT:
                        if isImplication(keyCmpElements):
                            if copKeyComparisonOutcome == CasusHappening.EQUIVALENT:
                                if CasusHappening.INDIFFERENT in set(keyCmp.values()):
                                    val = CasusHappening.INDIFFERENT
                                elif rhs_obligation_unmet:
                                    val = CasusHappening.INDIFFERENT
                                elif CasusHappening.LOSE_SPEC_IMPLICATION in set(keyCmp.values()):
                                    val = CasusHappening.INDIFFERENT
                                elif keyCmpElements == CasusHappening.INSTANTIATION_IMPLICATION or CasusHappening.INSTANTIATION_IMPLICATION in set(
                                        keyCmp.values()):
                                    val = keyCmpElements
                                else:
                                    val = CasusHappening.GENERAL_IMPLICATION
                            else:
                                val = keyCmpElements
                        elif keyCmpElementsInv == CasusHappening.LOSE_SPEC_IMPLICATION:
                            val = CasusHappening.INSTANTIATION_IMPLICATION
                        else:
                            val = keyCmpElements
                    else:
                        val = keyCmpElements
                elif isImplication(val):
                    if hasDirectSubset and keyCmpElements == CasusHappening.INDIFFERENT:
                        val = CasusHappening.INDIFFERENT
                    elif keyCmpElements != CasusHappening.EQUIVALENT:
                        if CasusHappening.INDIFFERENT in keyCmp.values():
                            val = CasusHappening.INDIFFERENT
                        elif rhs_obligation_unmet:
                            val = CasusHappening.INDIFFERENT
                        elif keyCmpElementsInv == CasusHappening.LOSE_SPEC_IMPLICATION:
                            val = CasusHappening.INSTANTIATION_IMPLICATION
                        else:
                            val = keyCmpElements
                    elif rhs_obligation_unmet:
                        val = CasusHappening.INDIFFERENT
                elif val == CasusHappening.EXCLUSIVES:
                    if (keyCmpElements == CasusHappening.INDIFFERENT) or (keyCmpElements == CasusHappening.EXCLUSIVES) or hasDirectSubset:
                        val = CasusHappening.INDIFFERENT
    if store:
        d[(x, y)] = val
    return val

def instantiate_rules(constituents, expansion_dictionary, final_constituents, isImpl):
    ls = list(reversed(constituents))
    result_list = list()
    from tqdm import tqdm
    
    label = "implication" if isImpl else "equivalence"
    pbar = tqdm(ls, desc=f"Expanding {label} constituents", leave=False)
    
    for original, (idx, constituent) in enumerate(pbar):
        str1 = str(constituent)
        from LaSSI.structures.extended_fol.TBoxReasoning import TBoxReasoningSingleton
        entrypoint, adj_graph, id_to_constituent, s = TBoxReasoningSingleton.explained_knowledge_expand(constituent, isImpl)
        # s.add(constituent)
        str2 = str(ls[original][1])
        if (str1 != str2):
            raise RuntimeError(str1+"!="+str2)
        expansion_dictionary[ls[original][1]] = s
        result = {"original": original,
                  "idx": idx,
                  "constituents": constituent,
                  "entrypoint": entrypoint,
                  "adj_graph": adj_graph,
                  "id_to_constituent": id_to_constituent}
        result_list.append(result)
    for y in expansion_dictionary.values():
        final_constituents.update(y)
    return result_list
    # return {(x, y): CasusHappening.NONE for x in final_constituents for y in
    #         final_constituents}


class ExpandConstituents:
    def __init__(self, cache_folder, constituents):
        """
        This class provides the expansion for each of the sentences, as well as caching the direction of the implication for each of the formulae
        """
        print("Setting up the rule expander...")
        from LaSSI.external_services.Services import Services
        # self.kb = kb

        self.constituents = constituents#list(constituents)
        _ied = os.path.join(cache_folder, "_ied.pickle")
        _ic = os.path.join(cache_folder, "_ic.pickle")
        _eed = os.path.join(cache_folder, "_eed.pickle")
        _ec = os.path.join(cache_folder, "_ec.pickle")
        explain_eq = os.path.join(cache_folder, "explain_eq.json")
        explain_impl = os.path.join(cache_folder, "explain_impl.json")
        # _exp = TBoxReasoningSingleton.get_ke_file_name()

        if (os.path.exists(explain_impl) and os.path.exists(explain_eq) and os.path.exists(_ied) and os.path.exists(_ic) and os.path.exists(_eed) and os.path.exists(_ec)):# and os.path.exists(_exp)
            with open(_ied, "rb") as f:
                self.impl_expansion_dictionary = pickle.load(f)
            with open(_ic, "rb") as f:
                self.impl_constituents = pickle.load(f)
            with open(_eed, "rb") as f:
                self.eq_expansion_dictionary = pickle.load(f)
            with open(_ec, "rb") as f:
                self.eq_constituents = pickle.load(f)

        else:
            self.impl_expansion_dictionary = dict()
            self.impl_constituents = set()
            self.eq_expansion_dictionary = dict()
            self.eq_constituents = set()

            # if not all(map(lambda x: isinstance(x, FBinaryPredicate) or isinstance(x, FUnaryPredicate),
            #                map(lambda x: x[1], self.constituents))):
            #     raise ValueError(
            #         "Error: all the rules within the set of rules must represent Predicates to be assessed, be them unary or binary")

            # Expanding the constituents

            Services.getInstance().log("Expanding the constituents...")
            # self.outcome_implication_dictionary =
            self.eq_explained = instantiate_rules(self.constituents, self.eq_expansion_dictionary, self.eq_constituents,
                              False)
            from LaSSI.files.JSONDump import json_dumps

            with open(explain_eq, "w") as f:
                f.write(json_dumps(self.eq_explained))
            self.impl_explained = instantiate_rules(self.constituents, self.impl_expansion_dictionary, self.impl_constituents,
                              True)
            with open(explain_impl, "w") as f:
                f.write(json_dumps(self.impl_explained))
            # self.outcome_eq_dictionary =

            with open(_ied, "wb") as f:
                pickle.dump(self.impl_expansion_dictionary, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open(_ic, "wb") as f:
                pickle.dump(self.impl_constituents, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open(_eed, "wb") as f:
                pickle.dump(self.eq_expansion_dictionary, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open(_ec, "wb") as f:
                pickle.dump(self.eq_constituents, f, protocol=pickle.HIGHEST_PROTOCOL)

        from LaSSI.explainer.FullExplainer import load_expansion_graph_from_json_file
        self.impl_explained = load_expansion_graph_from_json_file(explain_impl)
        self.eq_explained = load_expansion_graph_from_json_file(explain_eq)
        self.result_cache = dict()
        self.result_cache_raw = dict()
        self.ms = ModelSearch()
        self.lhsOrigDict = dict()
        self.rhsOrigDict = dict()
        self.inv_idx = dict()
        Services.getInstance().log("Splitting across unary and binary constituents for each sentence...")
        from tqdm import tqdm
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import time

        n = len(self.constituents)

        def _build_basis(item):
            row_i, row_sentence = item
            t0 = time.time()
            lhs = ModelSearchBasis(row_sentence, self.impl_expansion_dictionary[row_sentence])
            t_lhs = time.time() - t0
            t1 = time.time()
            rhs = ModelSearchBasis(row_sentence, self.eq_expansion_dictionary[row_sentence])
            t_rhs = time.time() - t1
            return row_i, row_sentence, lhs, rhs, t_lhs, t_rhs

        max_workers = min(n, os.cpu_count() or 4, 8)
        slowest = (-1.0, None)  # (elapsed, row_i)
        total_elapsed = 0.0
        completed = 0
        with tqdm(total=n, desc="Splitting constituents", unit="sent") as pbar:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(_build_basis, item): item for item in self.constituents}
                for future in as_completed(futures):
                    row_i, row_sentence, lhs, rhs, t_lhs, t_rhs = future.result()
                    self.inv_idx[row_sentence] = row_i
                    self.lhsOrigDict[row_i] = lhs
                    self.rhsOrigDict[row_i] = rhs
                    elapsed = t_lhs + t_rhs
                    total_elapsed += elapsed
                    completed += 1
                    if elapsed > slowest[0]:
                        slowest = (elapsed, row_i)
                    pbar.set_postfix({
                        "last_row": row_i,
                        "last": f"{elapsed:.2f}s",
                        "lhs/rhs": f"{t_lhs:.2f}/{t_rhs:.2f}",
                        "avg": f"{total_elapsed / completed:.2f}s",
                        "slowest": f"row{slowest[1]}@{slowest[0]:.1f}s",
                    })
                    pbar.update(1)
        Services.getInstance().log(
            f"  Splitting constituents done: {completed} sentences in {total_elapsed:.1f}s "
            f"(avg {total_elapsed / max(completed, 1):.2f}s/sent, slowest row {slowest[1]} @ {slowest[0]:.1f}s)"
        )
        self.constituents = dict(self.constituents)

    def getImplExpansions(self, idx):
        return self.lhsOrigDict[idx].all() if idx in self.lhsOrigDict else []

    def getEqExpansions(self, idx):
        return self.rhsOrigDict[idx].all() if idx in self.lhsOrigDict else []

    def getConstituentIDX(self, ith):
        from LaSSI.structures.extended_fol.TBoxReasoning import TBoxReasoningSingleton
        return TBoxReasoningSingleton.getConstituentIdx(ith)

    def getIthExpandedConstituent(self, ith):
        from LaSSI.structures.extended_fol.TBoxReasoning import TBoxReasoningSingleton
        return TBoxReasoningSingleton.getConstituentFromIdx(ith)

    def getImplExpansionExplanation(self, constituent):
        # constituent = self.constituents[idx]
        # assert idx == self.inv_idx[constituent]
        from LaSSI.structures.extended_fol.TBoxReasoning import TBoxReasoningSingleton
        return TBoxReasoningSingleton.subGraphImpl(constituent)

    def getEqExpansionExplanation(self, constituent):
        # constituent = self.constituents[idx]
        # assert idx == self.inv_idx[constituent]
        from LaSSI.structures.extended_fol.TBoxReasoning import TBoxReasoningSingleton
        return TBoxReasoningSingleton.subGraphEq(constituent)

    def determine_raw(self, i: int, j: int, forceEquiv:bool=False, isLeftDrop = False, isRightDrop = False):
        assert i in self.constituents
        assert j in self.constituents
        lhsOrig = self.lhsOrigDict[i]
        rhsOrig = self.lhsOrigDict[j] if forceEquiv else self.rhsOrigDict[j]
        tmp = self.ms.compare(lhsOrig, rhsOrig, isLeftDrop, isRightDrop)
        self.result_cache_raw[(i, j)] = tmp
        return tmp

    @staticmethod
    def rectify_implication(tmp):
        if tmp == CasusHappening.EXCLUSIVES:
            return PairwiseCases.ConflictingImplication
        elif tmp == CasusHappening.EQUIVALENT:
            return PairwiseCases.Equivalent
        elif isImplication(tmp):
            return PairwiseCases.Implying
        else:
            return PairwiseCases.Indifferent

    def determine(self, i: int, j: int):
        if (i == j):
            self.result_cache[(i, j)] = PairwiseCases.Equivalent
        assert i in self.constituents
        assert j in self.constituents
        # if (i, j) in self.result_cache:
        #     return self.result_cache[(i, j)]
        val = PairwiseCases.Indifferent

        lhsOrig = self.lhsOrigDict[i]
        rhsOrig = self.rhsOrigDict[j]
        tmp = self.ms.compare(lhsOrig, rhsOrig)
        if tmp == CasusHappening.EXCLUSIVES:
            val = PairwiseCases.ConflictingImplication
        elif tmp == CasusHappening.EQUIVALENT:
            val = PairwiseCases.Equivalent
        elif isImplication(tmp):
            val = PairwiseCases.Implying
        else:
            val = PairwiseCases.Indifferent

        self.result_cache[(i, j)] = val
        return val

    def getIDXGraph(self):
        from LaSSI.structures.extended_fol.TBoxReasoning import TBoxReasoningSingleton
        return TBoxReasoningSingleton.getIDXGraph()

    def getConstituentFromIDX(self, idx):
        from LaSSI.structures.extended_fol.TBoxReasoning import TBoxReasoningSingleton
        return TBoxReasoningSingleton.getConstituentFromIdx(idx)
