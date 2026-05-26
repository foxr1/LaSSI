from LaSSI.HOnK.HOnK import CasusHappening, HOnKSingleton
from LaSSI.structures.extended_fol.Enums import PairwiseCases
from LaSSI.structures.extended_fol.Formulae import FVariable, FNot, FBinaryPredicate, FUnaryPredicate, FAnd, FOr
from LaSSI.HOnK.TBox.ComparatorUtils import isImplication, transformCaseWhenOneArgIsNegated, isExistential
from LaSSI.HOnK.TBox.ParaphraseManager import _paraphrase_concept_of, _paraphrase_match
from LaSSI.HOnK.TBox.SpatialReasoner import _canonicalize_geo_fvar, _canonicalize_proper_noun_modifier, _strip_geo_generic_suffix, _has_near_place, _space_mismatch_contradiction, _space_city_mismatch_contradiction, _GEO_TYPES
from LaSSI.HOnK.TBox.LifecycleManager import _lifecycle_partition_verdict
from LaSSI.utils.datetime_canon import canonicalize_datetime_string as _canonicalize_datetime_string

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
    elif CasusHappening.EQUIVALENT in constituentCollection:
        return CasusHappening.EQUIVALENT
    else:
        return CasusHappening.INDIFFERENT

def simplifyConstituents(constituentCollection):
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
        vals = t1 if isinstance(t1, tuple) else (t1,)
        for x in vals:
            d1.append((k,x))
    for k, t2 in kv2:
        vals = t2 if isinstance(t2, tuple) else (t2,)
        for x in vals:
            d2.append((k, x))
    return set(d1).issubset(set(d2))

def _is_change_of_state_relation(rel):
    if rel is None or not HOnKSingleton.isReady():
        return False
    try:
        change_verbs = HOnKSingleton.get().getChangeOfStateVerbs() or set()
    except Exception:
        return False
    return str(rel).lower() in {str(v).lower() for v in change_verbs}

def _compare_single_prop_val(d, lhs_val, rhs_val, lhs_parent_concept=None, rhs_parent_concept=None):
    if lhs_val == rhs_val:
        return CasusHappening.EQUIVALENT

    def _name_match(a, b):
        if a == b:
            return True
        a_name = a if isinstance(a, str) else (a.name if isinstance(a, FVariable) else None)
        b_name = b if isinstance(b, str) else (b.name if isinstance(b, FVariable) else None)
        if a_name is not None and b_name is not None and a_name.lower() == b_name.lower():
            return True
        return False

    if isinstance(rhs_val, FOr):
        if any(_name_match(lhs_val, arg) for arg in rhs_val.args):
            return CasusHappening.GENERAL_IMPLICATION
    if isinstance(lhs_val, FOr):
        if any(_name_match(rhs_val, arg) for arg in lhs_val.args):
            return CasusHappening.INDIFFERENT

    if isinstance(lhs_val, FVariable) and isinstance(rhs_val, FVariable):
        return compare_variable(d, lhs_val, rhs_val, lhs_parent_concept=lhs_parent_concept, rhs_parent_concept=rhs_parent_concept)
    if isinstance(lhs_val, str) and isinstance(rhs_val, str):
        if lhs_val.lower() == rhs_val.lower():
            return CasusHappening.EQUIVALENT
        if _paraphrase_match(lhs_val, rhs_val, lhs_parent_concept=lhs_parent_concept, rhs_parent_concept=rhs_parent_concept):
            return CasusHappening.EQUIVALENT
        return HOnKSingleton.get().name_eq(lhs_val, rhs_val)
    if isinstance(lhs_val, str) and isinstance(rhs_val, FVariable):
        if rhs_val.name is not None and lhs_val.lower() == rhs_val.name.lower():
            return CasusHappening.EQUIVALENT
        return HOnKSingleton.get().name_eq(lhs_val, rhs_val.name)
    if isinstance(lhs_val, FVariable) and isinstance(rhs_val, str):
        if lhs_val.name is not None and lhs_val.name.lower() == rhs_val.lower():
            return CasusHappening.EQUIVALENT
        return HOnKSingleton.get().name_eq(lhs_val.name, rhs_val)
    return CasusHappening.INDIFFERENT

def _is_syntactic_fvar_property_key(key) -> bool:
    if isinstance(key, int):
        return True
    if isinstance(key, str) and key.isdigit():
        return True
    if key in {"det", "punct"}:
        return True
    return False

def _normalize_fvar_prop_key(k):
    if k in {"amod", "nummod", "cop", "JJ", "CD", "measurement"}:
        return "MODIFIER"
    return k

def _dict_from_props(props):
    d = {}
    for k, v in dict(props).items():
        if not _is_syntactic_fvar_property_key(k):
            nk = _normalize_fvar_prop_key(k)
            if nk not in d:
                d[nk] = []
            if isinstance(v, tuple):
                d[nk].extend(v)
            else:
                d[nk].append(v)
    return d

def _compare_fvar_properties(d, lhs_props, rhs_props, lhs_parent_concept=None, rhs_parent_concept=None):
    if lhs_props == rhs_props:
        return CasusHappening.EQUIVALENT
    lhs_dict = _dict_from_props(lhs_props)
    rhs_dict = _dict_from_props(rhs_props)
    all_keys = set(lhs_dict.keys()) | set(rhs_dict.keys())
    results = []
    for key in all_keys:
        if key in lhs_dict and key in rhs_dict:
            lhs_vals = lhs_dict[key]
            rhs_vals = rhs_dict[key]
            pair_results = [
                _compare_single_prop_val(d, lv, rv, lhs_parent_concept=lhs_parent_concept, rhs_parent_concept=rhs_parent_concept)
                for lv in lhs_vals for rv in rhs_vals
            ]
            results.append(simplifyConstituentsAcross(pair_results))
        elif key in lhs_dict:
            results.append(CasusHappening.GENERAL_IMPLICATION)
        else:
            results.append(CasusHappening.INDIFFERENT)
    return simplifyConstituentsAcross(results)

def _cop_parent_concept(v, fallback_concept):
    if isinstance(v, FVariable) and isinstance(v.specification, str):
        spec_concept = _paraphrase_concept_of(v.specification)
        if spec_concept is not None:
            return spec_concept
    return fallback_concept

def compare_variable(d, lhs, rhs, lhs_parent_concept=None, rhs_parent_concept=None):
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
    elif _paraphrase_match(lhs, rhs, lhs_parent_concept=lhs_parent_concept,
                           rhs_parent_concept=rhs_parent_concept):
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
        lhs = _canonicalize_geo_fvar(lhs)
        rhs = _canonicalize_geo_fvar(rhs)
        lhs = _canonicalize_proper_noun_modifier(lhs)
        rhs = _canonicalize_proper_noun_modifier(rhs)
        kb = HOnKSingleton.get()
        lhs_type = lhs.type or ""
        rhs_type = rhs.type or ""
        if (lhs.name is not None and rhs.name is not None
                and lhs.name != rhs.name and lhs.name.lower() == rhs.name.lower()):
            nameEQ = CasusHappening.EQUIVALENT
        else:
            nameEQ = kb.name_eq(lhs.name, rhs.name)
        if (nameEQ == CasusHappening.INDIFFERENT
                and lhs.name is not None and rhs.name is not None):
            lhs_dt = _canonicalize_datetime_string(lhs.name)
            if lhs_dt is not None:
                rhs_dt = _canonicalize_datetime_string(rhs.name)
                if rhs_dt is not None and lhs_dt == rhs_dt:
                    nameEQ = CasusHappening.EQUIVALENT
        if (nameEQ == CasusHappening.INDIFFERENT
                and lhs_type in _GEO_TYPES and rhs_type in _GEO_TYPES
                and lhs.name is not None and rhs.name is not None):
            lhs_stripped = _strip_geo_generic_suffix(lhs.name)
            rhs_stripped = _strip_geo_generic_suffix(rhs.name)
            if lhs_stripped != lhs.name or rhs_stripped != rhs.name:
                stripped_eq = kb.name_eq(lhs_stripped, rhs_stripped)
                if stripped_eq == CasusHappening.EQUIVALENT:
                    if _has_near_place(lhs) and _has_near_place(rhs):
                        nameEQ = CasusHappening.EQUIVALENT
                    else:
                        nameEQ = CasusHappening.GENERAL_IMPLICATION
        if lhs.specification is not None and rhs.specification is not None and lhs.specification.lower() == rhs.specification.lower():
            specEQ = CasusHappening.EQUIVALENT
        else:
            specEQ = kb.name_eq(lhs.specification, rhs.specification)
        specEQInv = kb.name_eq(rhs.specification, lhs.specification)
        if lhs.spec_negation != rhs.spec_negation:
            specEQ = transformCaseWhenOneArgIsNegated(specEQ)
        _lhs_concept = _paraphrase_concept_of(lhs)
        _rhs_concept = _paraphrase_concept_of(rhs)
        _lhs_cop_parent_concept = _cop_parent_concept(lhs, _lhs_concept)
        _rhs_cop_parent_concept = _cop_parent_concept(rhs, _rhs_concept)
        copCompareInv = compare_variable(d, rhs.cop, lhs.cop,
                                         lhs_parent_concept=_rhs_cop_parent_concept,
                                         rhs_parent_concept=_lhs_cop_parent_concept)
        val = CasusHappening.INDIFFERENT
        if (nameEQ == specEQ) and (specEQ == copCompareInv) and (lhs.asAll == rhs.asAll):
            if lhs.properties == rhs.properties:
                d[cp] = specEQ
                return d[cp]
            prop_cmp = _compare_fvar_properties(d, lhs.properties, rhs.properties,
                                                lhs_parent_concept=_lhs_concept,
                                                rhs_parent_concept=_rhs_concept)
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
                    nameAgainstSpec    = kb.name_eq(rhs.name, lhs.specification)
                    nameAgainstSpecInv = kb.name_eq(lhs.name, rhs.specification)
                    if (nameAgainstSpec == CasusHappening.EQUIVALENT
                            and rhs.name is not None and lhs.specification is not None) \
                       or (nameAgainstSpecInv == CasusHappening.EQUIVALENT
                            and lhs.name is not None and rhs.specification is not None):
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
                val = nameEQ if (not rhs.asAll) and lhs.asAll else CasusHappening.INDIFFERENT
            elif (specEQ == CasusHappening.EQUIVALENT) and (copCompareInv == CasusHappening.EXCLUSIVES):
                val = CasusHappening.EXCLUSIVES
            elif lhs.specification is None and nameAgainstSpec == CasusHappening.EQUIVALENT:
                val = CasusHappening.INSTANTIATION_IMPLICATION if lhs.asAll else CasusHappening.INDIFFERENT
        elif nameEQ == CasusHappening.EXCLUSIVES:
            if (specEQ == copCompareInv) and (specEQ == CasusHappening.EQUIVALENT):
                val = CasusHappening.EXCLUSIVES
    if val == CasusHappening.INDIFFERENT:
        if _paraphrase_match(lhs, rhs):
            val = CasusHappening.EQUIVALENT
    d[cp] = val
    return d[cp]

def test_pairwise_sentence_similarity(d, x, y, store=True, shift=True):
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

            if (is_direct_subset(xprop, yprop) and len(xprop)>0) or (len(yprop) == 0 and len(xprop) > 0):
                keyCmpElements = CasusHappening.GENERAL_IMPLICATION
                keyCmpElementsInv = CasusHappening.INDIFFERENT
                hasDirectSubset = True
            else:
                for key in keys:
                    if key in dLHS and key in dRHS:
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
                        keyCmp[key] = CasusHappening.GENERAL_IMPLICATION
                        keyCmpInv[key] = CasusHappening.INSTANTIATION_IMPLICATION if soft else CasusHappening.INDIFFERENT
                    else:
                        soft = any(_value_in_other(yy, dLHS) for yy in dRHS[key])
                        keyCmp[key] = CasusHappening.INSTANTIATION_IMPLICATION if soft else CasusHappening.INDIFFERENT
                        keyCmpInv[key] = CasusHappening.GENERAL_IMPLICATION
                if len(keyCmp) > 0:
                    keyCmpElements = simplifyConstituentsAcross({keyCmp[key] for key in keyCmp})
                    keyCmpElementsInv = simplifyConstituentsAcross({keyCmpInv[key] for key in keyCmpInv})
                else:
                    keyCmpElements, keyCmpElementsInv = CasusHappening.EQUIVALENT, CasusHappening.EQUIVALENT

            # FIX: Ensure these are always bound
            keyComparisonOutcome = CasusHappening.INDIFFERENT
            copKeyComparisonOutcome = CasusHappening.INDIFFERENT

            antonymRelationContradiction = False
            _lifecycle_verdict = _lifecycle_partition_verdict(x, y)
            _space_mismatch = _space_mismatch_contradiction(x, y)
            _space_city_mismatch = _space_city_mismatch_contradiction(x, y)
            _causation_mismatch = False

            if (_lifecycle_verdict == 'contradiction'
                    or _space_mismatch
                    or _space_city_mismatch
                    or _causation_mismatch):
                val = CasusHappening.EXCLUSIVES
                antonymRelationContradiction = True
            elif isinstance(x, FBinaryPredicate) and isinstance(y, FBinaryPredicate):
                relCmp = HOnKSingleton.get().name_eq(x.rel, y.rel) if x.rel != y.rel else CasusHappening.EQUIVALENT
                if relCmp == CasusHappening.INDIFFERENT:
                    val = CasusHappening.INDIFFERENT
                else:
                    srcCmp = compare_variable(d, x.src, y.src)
                    dstCmp = compare_variable(d, x.dst, y.dst)
                    if relCmp == CasusHappening.EXCLUSIVES:
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
                pass
            elif val != CasusHappening.INDIFFERENT:
                rhs_obligation_unmet = False
                for k in dRHS:
                    if k not in dLHS:
                        if not any(_value_in_other(yy, dLHS) for yy in dRHS[k]):
                            rhs_obligation_unmet = True
                            break
                    else:
                        for yy in dRHS[k]:
                            unmet = True
                            for xx in dLHS[k]:
                                cv_fwd = compare_variable(d, xx, yy)
                                cv_rev = compare_variable(d, yy, xx)
                                if cv_fwd not in (CasusHappening.INDIFFERENT, CasusHappening.EXCLUSIVES) or \
                                        cv_rev not in (CasusHappening.INDIFFERENT, CasusHappening.EXCLUSIVES):
                                    unmet = False
                                    break
                            if unmet:
                                rhs_obligation_unmet = True
                                break
                    if rhs_obligation_unmet:
                        break
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
