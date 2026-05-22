from LaSSI.structures.extended_fol.Formulae import FVariable, FOr, FUnaryPredicate, FBinaryPredicate
from LaSSI.HOnK.HOnK import HOnKSingleton, CasusHappening
from LaSSI.HOnK.TBox.ComparatorUtils import isImplication

_GEO_TYPES = {"LOC", "GPE", "FAC"}
_SPATIAL_MOVEMENT_LABELS = frozenset({"stay in place", "motion to place", "motion from place"}) # TODO: modify this to take from logical_analysis.json
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
