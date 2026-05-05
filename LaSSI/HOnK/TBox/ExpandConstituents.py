import copy
import json
import os.path
import pickle
from collections import defaultdict

from torch.fx.experimental.symbolic_shapes import lru_cache

from LaSSI.structures.extended_fol.Enums import PairwiseCases
from LaSSI.structures.extended_fol.ModelSearch import ModelSearch, ModelSearchBasis
from LaSSI.structures.extended_fol.TBoxReasoning import TBoxReasoningSingleton
from LaSSI.structures.extended_fol.Formulae import FVariable, FNot, FBinaryPredicate, FUnaryPredicate
from LaSSI.HOnK.HOnK import CasusHappening, HOnKSingleton

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


# Paraphrase concepts: surface-distinct values that denote the same idea.
# Each entry maps a concept tag to a set of (form, name) pairs:
#   ("name", n)      → matches FVariable.name == n
#   ("fnot_name", n) → matches FNot(FVariable(name=n, ...))
# Two values with the same concept tag are treated as EQUIVALENT by
# `compare_variable`, even when WordNet/HOnK does not relate them.
PARAPHRASE_CONCEPTS = {
    "indefinite_suspension": {
        ("name", "until further notice"),
        ("name", "indefinitely"),
        ("fnot_name", "reopening"),
        ("fnot_name", "resumption"),
        ("fnot_name", "restart"),
    },
}


def _paraphrase_concept_of(v):
    if isinstance(v, FVariable) and v.name:
        nm = v.name.strip().lower()
        for concept, members in PARAPHRASE_CONCEPTS.items():
            if ("name", nm) in members:
                return concept
    elif isinstance(v, FNot):
        inner = v.arg
        if isinstance(inner, FVariable) and inner.name:
            nm = inner.name.strip().lower()
            for concept, members in PARAPHRASE_CONCEPTS.items():
                if ("fnot_name", nm) in members:
                    return concept
    return None


def _paraphrase_match(lhs, rhs):
    cl = _paraphrase_concept_of(lhs)
    if cl is None:
        return False
    return cl == _paraphrase_concept_of(rhs)


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
        kb = HOnKSingleton.get()
        nameEQ = kb.name_eq(lhs.name, rhs.name)
        specEQ = kb.name_eq(lhs.specification, rhs.specification)
        specEQInv = kb.name_eq(rhs.specification, lhs.specification)
        if lhs.spec_negation != rhs.spec_negation:
            specEQ = transformCaseWhenOneArgIsNegated(specEQ)
        copCompareInv = compare_variable(d, rhs.cop, lhs.cop)
        val = CasusHappening.INDIFFERENT
        if (nameEQ == specEQ) and (specEQ == copCompareInv) and (lhs.asAll == rhs.asAll):
            d[cp] = specEQ
            return d[cp]
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
            keyCmp, keyCmpInv = defaultdict(set), defaultdict(set)
            keys = set(map(lambda z: z[0], xprop)).union(map(lambda z: z[0], yprop))
            hasDirectSubset = False
            dLHS = dict(xprop)
            dRHS = dict(yprop)
            if (is_direct_subset(xprop, yprop) and len(xprop)>0) or (len(yprop) == 0 and len(xprop) > 0):
                keyCmpElements = CasusHappening.GENERAL_IMPLICATION
                keyCmpElementsInv = CasusHappening.INDIFFERENT
                hasDirectSubset = True
            elif set(dLHS.keys()).issubset(set(dRHS.keys())) and set(dLHS.keys()) != set(dRHS.keys()):
                keyCmpElements = CasusHappening.INDIFFERENT
                keyCmpElementsInv = CasusHappening.INDIFFERENT
                hasDirectSubset = True
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
                            if compare_variable(d, val, ov) == CasusHappening.EQUIVALENT:
                                return True
                    return False

                for key in keys:
                    if key in dLHS and key in dRHS:
                        for xx in dLHS[key]:
                            for yy in dRHS[key]:
                                keyCmp[key].add(compare_variable(d, xx, yy))
                                keyCmpInv[key].add(compare_variable(d, yy, xx))
                    elif key in dLHS:
                        # Same value under a different RHS key counts as a
                        # role-mismatch soft match (INSTANTIATION_IMPLICATION)
                        # rather than INDIFFERENT.  Sibling-key alone (no
                        # value overlap) stays INDIFFERENT — different roles
                        # with different content shouldn't claim equivalence.
                        soft = any(_value_in_other(xx, dRHS) for xx in dLHS[key])
                        keyCmp[key].add(CasusHappening.INSTANTIATION_IMPLICATION
                                        if soft else CasusHappening.INDIFFERENT)
                        keyCmpInv[key].add(CasusHappening.GENERAL_IMPLICATION)
                    else:
                        soft = any(_value_in_other(yy, dLHS) for yy in dRHS[key])
                        keyCmp[key].add(CasusHappening.GENERAL_IMPLICATION)
                        keyCmpInv[key].add(CasusHappening.INSTANTIATION_IMPLICATION
                                           if soft else CasusHappening.INDIFFERENT)
                keyCmp = {key: simplifyConstituents(val) for key, val in keyCmp.items()}
                keyCmpInv = {key: simplifyConstituents(val) for key, val in keyCmpInv.items()}
                if len(keyCmp) > 0:
                    keyCmpElements = simplifyConstituentsAcross({keyCmp[key] for key in keyCmp})
                    keyCmpElementsInv = simplifyConstituentsAcross({keyCmpInv[key] for key in keyCmpInv})
                else:
                    keyCmpElements, keyCmpElementsInv = CasusHappening.EQUIVALENT, CasusHappening.EQUIVALENT
            antonymRelationContradiction = False
            if isinstance(x, FBinaryPredicate) and isinstance(y, FBinaryPredicate):
                # Compare relation names through the ontology, not by string
                # equality — so antonyms like close/open trigger EXCLUSIVES
                # propagation when both arguments coincide.
                relCmp = HOnKSingleton.get().name_eq(x.rel, y.rel) if x.rel != y.rel else CasusHappening.EQUIVALENT
                if relCmp == CasusHappening.INDIFFERENT:
                    val = CasusHappening.INDIFFERENT
                else:
                    srcCmp = compare_variable(d, x.src, y.src)
                    if (srcCmp == CasusHappening.INDIFFERENT):
                        val = CasusHappening.INDIFFERENT
                    else:
                        dstCmp = compare_variable(d, x.dst, y.dst)
                        if (dstCmp == CasusHappening.INDIFFERENT):
                            val = CasusHappening.INDIFFERENT
                        else:
                            keyComparisonOutcome = compare_variable(d, x.src, y.src)
                            copKeyComparisonOutcome = compare_variable(d, x.src.cop if hasattr(x.src, "cop") else None, y.src.cop if hasattr(y.src, "cop") else None)
                            if relCmp == CasusHappening.EXCLUSIVES:
                                # Antonym predicate: contradiction iff both arguments are equivalent.
                                if srcCmp == CasusHappening.EQUIVALENT and dstCmp == CasusHappening.EQUIVALENT:
                                    val = CasusHappening.EXCLUSIVES
                                    antonymRelationContradiction = True
                                else:
                                    val = CasusHappening.INDIFFERENT
                            elif (srcCmp == CasusHappening.EXCLUSIVES) and (dstCmp == CasusHappening.EXCLUSIVES):
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
                if val == CasusHappening.EQUIVALENT:
                    if keyComparisonOutcome == CasusHappening.EQUIVALENT:
                        if isImplication(keyCmpElements):
                            if copKeyComparisonOutcome == CasusHappening.EQUIVALENT:
                                if CasusHappening.INDIFFERENT in set(keyCmp.values()):
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
                    if keyCmpElements != CasusHappening.EQUIVALENT:
                        if CasusHappening.INDIFFERENT in keyCmp.values():
                            val = CasusHappening.INDIFFERENT
                        elif keyCmpElementsInv == CasusHappening.LOSE_SPEC_IMPLICATION:
                            val = CasusHappening.INSTANTIATION_IMPLICATION
                        else:
                            val = keyCmpElements
                elif val == CasusHappening.EXCLUSIVES:
                    if (keyCmpElements == CasusHappening.INDIFFERENT) or (keyCmpElements == CasusHappening.EXCLUSIVES) or hasDirectSubset:
                        val = CasusHappening.INDIFFERENT
    if store:
        d[(x, y)] = val
    return val

def instantiate_rules(constituents, expansion_dictionary, final_constituents, isImpl):
    ls = list(reversed(constituents))
    result_list = list()
    from LaSSI.external_services.Services import Services
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

        n = len(self.constituents)

        def _build_basis(item):
            row_i, row_sentence = item
            lhs = ModelSearchBasis(row_sentence, self.impl_expansion_dictionary[row_sentence])
            rhs = ModelSearchBasis(row_sentence, self.eq_expansion_dictionary[row_sentence])
            return row_i, row_sentence, lhs, rhs

        max_workers = min(n, os.cpu_count() or 4, 8)
        with tqdm(total=n, desc="Splitting constituents", unit="sent") as pbar:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(_build_basis, item): item for item in self.constituents}
                for future in as_completed(futures):
                    row_i, row_sentence, lhs, rhs = future.result()
                    self.inv_idx[row_sentence] = row_i
                    self.lhsOrigDict[row_i] = lhs
                    self.rhsOrigDict[row_i] = rhs
                    pbar.update(1)
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
