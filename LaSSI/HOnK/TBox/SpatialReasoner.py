from LaSSI.structures.extended_fol.Formulae import FVariable, FOr, FAnd, FUnaryPredicate, FBinaryPredicate, FNot
from LaSSI.HOnK.HOnK import HOnKSingleton, CasusHappening
from LaSSI.HOnK.TBox.ComparatorUtils import isImplication

_GEO_TYPES = {"LOC", "GPE", "FAC"}
_SPATIAL_MOVEMENT_LABELS = frozenset({"stay in place", "motion to place", "motion from place"}) # TODO: modify this to take from logical_analysis.json
# The full set of spatial-relation type labels (movement labels plus the
# proximity label "near place") that can appear under a SPACE value's `type`
# property. Used to compute the directional implication between e.g. a plain
# "near place" and a disjunctive "on or near" = OR(stay in place, near place).
_SPATIAL_RELATION_LABELS = _SPATIAL_MOVEMENT_LABELS | frozenset({"near place"})
_PROPER_NOUN_TYPES = {"LOC", "GPE", "ORG", "PERSON", "FAC"}

def _strip_spatial_type_properties(props):
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

def _canonicalize_proper_noun_modifier(fvar):
    if not isinstance(fvar, FVariable):
        return fvar
    if (fvar.type or "").upper() not in _PROPER_NOUN_TYPES:
        return fvar
    spec = fvar.specification
    if not isinstance(spec, str) or not spec.strip():
        return fvar
    spec_first = spec.strip().split()[0].lower()
    if spec_first in HOnKSingleton.get().getFacilityNouns():
        return fvar
    return FVariable(spec, "noun", None, None, fvar.id, fvar.properties,
                     fvar.spec_negation, fvar.meta, fvar.asAll)

def _strip_geo_generic_suffix(name: str | None) -> str | None:
    if name is None:
        return None
    words = name.strip().split()
    if len(words) > 1 and words[-1].lower() in HOnKSingleton.get().getGeoSuffixNouns():
        return " ".join(words[:-1])
    return name

def _has_near_place(fvar) -> bool:
    if not isinstance(fvar, FVariable) or not fvar.properties:
        return False
    for key, val in fvar.properties:
        if key != "type":
            continue
        vals = val if isinstance(val, tuple) else (val,)
        for v in vals:
            if isinstance(v, FOr):
                for arg in getattr(v, "args", ()) or ():
                    arg_name = getattr(arg, "name", None)
                    if arg_name and "near place" in arg_name.lower():
                        return True
                continue
            v_str = v if isinstance(v, str) else getattr(v, "name", None)
            if v_str and "near place" in v_str.lower():
                return True
    return False

def _space_near_place_names(formula):
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
            if isinstance(t, FOr):
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
            stripped = _strip_geo_generic_suffix(v.name)
            names.add(stripped.strip())
    return names if names else None

def _space_mismatch_contradiction(lhs, rhs) -> bool:
    # If the two SPACE descriptors corefer (e.g. one names a place as a single
    # "<Locality> <Facility>" compound and the other decomposes it into a
    # [<Facility>, <Locality>] pair), they denote the same place and cannot be
    # a contradiction — regardless of the spatial-relation direction.
    if _space_coreference_verdict(lhs, rhs) is not None:
        return False
    lhs_names = _space_near_place_names(lhs)
    rhs_names = _space_near_place_names(rhs)
    if not lhs_names or not rhs_names:
        return False
    if lhs_names & rhs_names:
        return False
    if {x.lower() for x in lhs_names} & {x.lower() for x in rhs_names}:
        return False
    if not HOnKSingleton.isReady():
        return True
    kb = HOnKSingleton.get()
    for ln in lhs_names:
        for rn in rhs_names:
            if _geo_names_compatible(kb, ln, rn):
                return False
    return True

def _space_named_geo_entities(formula):
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
    out = set()
    for v in space_vals:
        if not isinstance(v, FVariable):
            continue
        if v.name is None or (isinstance(v.name, str) and v.name.startswith("?")):
            continue
        if (v.type or "").upper() not in _GEO_TYPES:
            continue
        if v.properties:
            is_near = False
            for kk, vv in v.properties:
                if kk != "type":
                    continue
                type_vals = vv if isinstance(vv, tuple) else (vv,)
                for t in type_vals:
                    t_str = t if isinstance(t, str) else (t.name if isinstance(t, FVariable) else None)
                    if t_str and "near place" in str(t_str).lower():
                        is_near = True
                        break
                if is_near:
                    break
            if is_near:
                continue
        out.add(v.name.strip())
    return out if out else None

def _space_city_mismatch_contradiction(lhs, rhs) -> bool:
    if _space_coreference_verdict(lhs, rhs) is not None:
        return False
    lhs_names = _space_named_geo_entities(lhs)
    rhs_names = _space_named_geo_entities(rhs)
    if not lhs_names or not rhs_names:
        return False
    if lhs_names & rhs_names:
        return False
    if {x.lower() for x in lhs_names} & {x.lower() for x in rhs_names}:
        return False
    if not HOnKSingleton.isReady():
        return True
    kb = HOnKSingleton.get()
    for ln in lhs_names:
        compatible = False
        for rn in rhs_names:
            if _geo_names_compatible(kb, ln, rn):
                compatible = True
                break
        if not compatible:
            return True
    return False

def _geo_names_compatible(kb, lhs_name: str, rhs_name: str) -> bool:
    """Compare location names exactly first, then case-fold as a fallback.

    Some ontology entries are case-sensitive enough that lowercasing first can
    lose a good match ("Newcastle upon Tyne" -> "Newcastle").  Other inputs
    differ only by parser casing ("Parking Area" vs "parking area").  Preserve
    the explicit form as authoritative, and use lowercase only when the exact
    comparison is inconclusive.
    """
    eq = kb.name_eq(lhs_name, rhs_name)
    if eq == CasusHappening.EQUIVALENT or isImplication(eq):
        return True
    lhs_lower = lhs_name.lower()
    rhs_lower = rhs_name.lower()
    if lhs_lower == rhs_lower:
        return True
    if lhs_lower != lhs_name or rhs_lower != rhs_name:
        eq_lower = kb.name_eq(lhs_lower, rhs_lower)
        if eq_lower == CasusHappening.EQUIVALENT or isImplication(eq_lower):
            return True
    return False


# --- Compound-vs-decomposed location coreference -------------------------
#
# A location is sometimes parsed as a single "<Locality> <Facility>" compound
# Singleton ("Newcastle Bus Station") and elsewhere as a [<Facility>, <Locality>]
# pair of Singletons ("Bus/Coach Station" + "Newcastle"). These denote the same
# real-world place. Recognising the coreference needs the city and facility
# parts *jointly*, so it can't be expressed as a per-FVariable comparison; the
# helpers below operate on the whole SPACE value-list of a predicate.
#
# Facility-head compatibility is data-driven from HOnK (synonymy + a single isA
# hop), and never a hardcoded word list. The directional verdict reflects the
# spatial-relation type (near vs on-or-near), so a plain "near" entails a
# disjunctive "on or near" but not vice versa.


def _space_geo_values(formula):
    """Return the list of FVariable values under the SPACE property key of a
    predicate (unwrapping FNot)."""
    if isinstance(formula, FNot):
        formula = formula.arg
    if not isinstance(formula, (FBinaryPredicate, FUnaryPredicate)):
        return []
    if not formula.properties:
        return []
    for k, vs in formula.properties:
        if k == "SPACE":
            vals = vs if isinstance(vs, tuple) else (vs,)
            return [v for v in vals if isinstance(v, FVariable)]
    return []


def _contains_class_noun(name) -> bool:
    """True iff `name` contains a HOnK-known facility / access-point / route /
    location class noun as a whitespace-delimited token (so "Newcastle Bus
    Station" matches via "station", but bare "Newcastle" does not). Sourced
    from the ontology noun lists, never hardcoded."""
    from LaSSI.ner.string_functions import _ontology_class_suffix_terms
    if not isinstance(name, str) or not name:
        return False
    terms = _ontology_class_suffix_terms()
    if not terms:
        return False
    lowered = name.lower()
    tokens = set(lowered.replace("/", " ").split())
    for term in terms:
        if " " in term:
            if term in lowered:
                return True
        elif term in tokens:
            return True
    return False


def _facility_head_compatible(kb, fa, fb) -> 'CasusHappening':
    """Compare two facility/location head phrases ("bus station" vs "bus/coach
    station", "coach station" vs "bus station"), bypassing name_eq's
    supertype-intersection gate (which short-circuits to INDIFFERENT when one
    side has an empty supertype closure). Returns EQUIVALENT when they denote
    the same facility type via case-insensitive equality, HOnK synonymy, or a
    single isA hop in either direction; otherwise INDIFFERENT.

    A sub/super-type facility within the *same locality* is taken to be the
    same physical place, so the verdict is symmetric EQUIVALENT — the only
    coreference asymmetry comes from the spatial-relation type, computed
    separately in `_space_coreference_verdict`."""
    if not fa or not fb:
        return CasusHappening.INDIFFERENT
    a = fa.strip().lower()
    b = fb.strip().lower()
    if not a or not b:
        return CasusHappening.INDIFFERENT
    if a == b:
        return CasusHappening.EQUIVALENT
    if not kb.getSynonymy(a).isdisjoint(kb.getSynonymy(b)):
        return CasusHappening.EQUIVALENT
    isA = getattr(kb, "_isA_supers", {})
    if b in isA.get(a, ()) or a in isA.get(b, ()):
        return CasusHappening.EQUIVALENT
    return CasusHappening.INDIFFERENT


def _decompose_compound(compound, locality):
    """If `locality`'s tokens are a contiguous prefix or suffix of `compound`'s
    tokens, return the remaining (facility) tokens as a string; else None.
    Handles both "Newcastle Bus Station" (city prefix) and "Bus/Coach Station
    Newcastle" (city suffix)."""
    if not compound or not locality:
        return None
    ct = compound.split()
    lt = locality.split()
    if len(ct) <= len(lt):
        return None
    lc = [t.lower() for t in ct]
    ll = [t.lower() for t in lt]
    if lc[:len(ll)] == ll:
        return " ".join(ct[len(ll):])
    if lc[-len(ll):] == ll:
        return " ".join(ct[:-len(ll)])
    return None


def _spaces_corefer(lhs_vals, rhs_vals, kb) -> bool:
    """True iff one SPACE list is a single "<Locality> <Facility>" compound and
    the other decomposes it into a separate locality + a facility whose head is
    ontologically compatible (the compound-vs-decomposed case)."""
    for comp_vals, other_vals in ((lhs_vals, rhs_vals), (rhs_vals, lhs_vals)):
        comp_names = [v.name for v in comp_vals
                      if v.name and _contains_class_noun(v.name)]
        other_names = [v.name for v in other_vals if v.name]
        other_facilities = [n for n in other_names if _contains_class_noun(n)]
        for cf in comp_names:
            for loc in other_names:
                remainder = _decompose_compound(cf, loc)
                if not remainder or not _contains_class_noun(remainder):
                    continue
                for fac in other_facilities:
                    if _facility_head_compatible(kb, remainder, fac) == CasusHappening.EQUIVALENT:
                        return True
    return False


def _collect_label_strings(t):
    """Lowercased label strings carried by a `type` value, expanding FOr arms
    (each arm is typically an FVariable named after the spatial relation)."""
    out = set()
    if isinstance(t, str):
        if t.strip():
            out.add(t.strip().lower())
    elif isinstance(t, FOr):
        for arg in getattr(t, "args", ()) or ():
            out |= _collect_label_strings(arg)
    elif isinstance(t, FVariable):
        if t.name and t.name.strip():
            out.add(t.name.strip().lower())
    return out


def _spatial_relation_labels(vals):
    """Aggregate the spatial-relation type labels (stay/near/motion) of the
    facility-bearing SPACE entities, expanding any FOr disjunction."""
    labels = set()
    for v in vals:
        if not isinstance(v, FVariable) or not v.properties or not v.name:
            continue
        if not _contains_class_noun(v.name):
            continue
        for k, val in v.properties:
            if k != "type":
                continue
            for t in (val if isinstance(val, tuple) else (val,)):
                labels |= _collect_label_strings(t)
    return labels & _SPATIAL_RELATION_LABELS


def _space_coreference_verdict(lhs_formula, rhs_formula):
    """Directional verdict for two predicates whose SPACE descriptors corefer
    via the compound-vs-decomposed pattern, or None when inapplicable (caller
    falls back to the default pairwise comparison — so unrelated pairs are
    unaffected).

    When the places corefer, the verdict reflects the spatial-relation type
    implication: a plain "near place" entails a disjunctive "on or near" =
    OR(stay in place, near place) (GENERAL_IMPLICATION), but not vice versa
    (INDIFFERENT); identical relation sets give EQUIVALENT."""
    if not HOnKSingleton.isReady():
        return None
    lhs_vals = _space_geo_values(lhs_formula)
    rhs_vals = _space_geo_values(rhs_formula)
    if not lhs_vals or not rhs_vals:
        return None
    kb = HOnKSingleton.get()
    if not _spaces_corefer(lhs_vals, rhs_vals, kb):
        return None
    lhs_rel = _spatial_relation_labels(lhs_vals)
    rhs_rel = _spatial_relation_labels(rhs_vals)
    if not lhs_rel or not rhs_rel or lhs_rel == rhs_rel:
        return CasusHappening.EQUIVALENT
    if lhs_rel <= rhs_rel:
        return CasusHappening.GENERAL_IMPLICATION
    return CasusHappening.INDIFFERENT


def _space_relation_narrowing(lhs_formula, rhs_formula) -> bool:
    """True when LHS only commits to a broader spatial relation than RHS.

    Example: a predicate located "on or near" a place carries
    OR(stay in place, near place), while another says simply "near place".
    The values are related enough to be partial support, but LHS should not
    fully entail RHS because "on" is also allowed.
    """
    for lhs_pred in _iter_predicate_atoms(lhs_formula):
        for rhs_pred in _iter_predicate_atoms(rhs_formula):
            if getattr(lhs_pred, "rel", None) != getattr(rhs_pred, "rel", None):
                continue
            for lhs_val in _space_geo_values(lhs_pred):
                lhs_rel = _spatial_relation_labels_for_value(lhs_val)
                if not lhs_rel:
                    continue
                for rhs_val in _space_geo_values(rhs_pred):
                    if not _space_values_name_compatible(lhs_val, rhs_val):
                        continue
                    rhs_rel = _spatial_relation_labels_for_value(rhs_val)
                    if rhs_rel and rhs_rel < lhs_rel:
                        return True
    return False


def _iter_predicate_atoms(formula):
    if isinstance(formula, FNot):
        formula = formula.arg
    if isinstance(formula, (FBinaryPredicate, FUnaryPredicate)):
        yield formula
        return
    if isinstance(formula, (FAnd, FOr)):
        for arg in getattr(formula, "args", ()) or ():
            yield from _iter_predicate_atoms(arg)


def _spatial_relation_labels_for_value(value):
    if not isinstance(value, FVariable) or not value.properties:
        return set()
    labels = set()
    for key, val in value.properties:
        if key != "type":
            continue
        for item in (val if isinstance(val, tuple) else (val,)):
            labels |= _collect_label_strings(item)
    return labels & _SPATIAL_RELATION_LABELS


def _space_values_name_compatible(lhs_val, rhs_val):
    lhs_name = getattr(lhs_val, "name", None)
    rhs_name = getattr(rhs_val, "name", None)
    if not lhs_name or not rhs_name:
        return False
    if lhs_name == rhs_name or lhs_name.lower() == rhs_name.lower():
        return True
    if not HOnKSingleton.isReady():
        return False
    return _geo_names_compatible(HOnKSingleton.get(), lhs_name, rhs_name)
