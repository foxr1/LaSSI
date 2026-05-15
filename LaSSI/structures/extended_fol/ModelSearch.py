# from LaSSI.HOnK.TBox.ExpandConstituents import CasusHappening, test_pairwise_sentence_similarity, isImplication
# from logical_repr.Sentences import FUnaryPredicate, FBinaryPredicate, FNot
# from logical_repr.rewrite_kernels import make_not
from pydatagramdb import result

from LaSSI.structures.extended_fol.Formulae import *
from LaSSI.HOnK.HOnK import CasusHappening

_TRACE_COMPARE = True  # toggle for debug tracing


class ModelSearchBasis:
    def __init__(self, original, constituents):
        self.original = original
        self.unary = []
        self.binary = []
        if isinstance(original, FUnaryPredicate):
            self.unary.insert(0, original)
        elif isinstance(original, FBinaryPredicate):
            self.binary.insert(0, original)
        else:
            raise Exception("Unexpected expression: " + str(original))
        from LaSSI.structures.extended_fol.TBoxReasoning import non_redundant_constituents
        for constituent in constituents:
            if isinstance(constituent, FUnaryPredicate) or (isinstance(constituent, FNot) and isinstance(constituent.arg, FUnaryPredicate)):
                if not (constituent == original) and non_redundant_constituents(constituent, False):
                    self.unary.append(constituent)
            elif isinstance(constituent, FBinaryPredicate) or (isinstance(constituent, FNot) and isinstance(constituent.arg, FBinaryPredicate)):
                if not (constituent == original) and non_redundant_constituents(constituent, False):
                    self.binary.append(constituent)
            else:
                raise Exception("Unexpected expression: "+str(constituent))

    def all(self):
        return self.unary + self.binary



class ModelSearch:
    def __init__(self):
        self.pairwise_similarity_cache = dict()
        # self.kb = kb
        self.main_cache = dict()

    @staticmethod
    def _same_relation(lhs, rhs):
        if not (isinstance(lhs, (FUnaryPredicate, FBinaryPredicate)) and
                isinstance(rhs, (FUnaryPredicate, FBinaryPredicate))):
            return False
        if lhs.rel == rhs.rel:
            return True
        # Synonymous relations (e.g. 'carry out' / 'do') should also count
        # so the contextual guard fires and re-checks the originals' properties.
        from LaSSI.HOnK.HOnK import HOnKSingleton
        from LaSSI.HOnK.TBox.ExpandConstituents import isImplication
        if HOnKSingleton.isReady():
            rel_cmp = HOnKSingleton.get().name_eq(lhs.rel, rhs.rel)
            if rel_cmp == CasusHappening.EQUIVALENT or isImplication(rel_cmp):
                return True
        return False

    @staticmethod
    def _has_properties(formula):
        return (
                isinstance(formula, (FUnaryPredicate, FBinaryPredicate)) and
                formula.properties is not None and len(formula.properties) > 0
        )

    def _guard_contextual_implication(self, objLHS, objRHS, verdict):
        from LaSSI.HOnK.TBox.ExpandConstituents import (
            test_pairwise_sentence_similarity, isImplication,
            compare_variable, simplifyConstituentsAcross, simplifyConstituents,
        )
        if not isImplication(verdict):
            return verdict
        # When either original carries properties (CAUSATION, SPACE, etc.),
        # re-check whether the originals' properties are compatible.
        # Expansion-derived implications strip optional properties and can
        # produce false positives when the originals disagree on those
        # properties (e.g. different causes).
        if not (self._has_properties(objLHS.original) or self._has_properties(objRHS.original)):
            if _TRACE_COMPARE:
                print(f"[GUARD] no properties on either original, returning verdict={verdict}")
            return verdict
        # First try a full direct comparison of the originals.  This covers
        # the common case where the relation names are string-equal or the
        # ontology recognises them as synonyms.
        if _TRACE_COMPARE:
            lhs_props = getattr(objLHS.original, 'properties', None)
            rhs_props = getattr(objRHS.original, 'properties', None)
            print(f"[GUARD] LHS original: rel={getattr(objLHS.original,'rel','?')}, dst={getattr(objLHS.original,'dst','?')}, props={lhs_props}")
            print(f"[GUARD] RHS original: rel={getattr(objRHS.original,'rel','?')}, dst={getattr(objRHS.original,'dst','?')}, props={rhs_props}")
            print(f"[GUARD] LHS str: {objLHS.original}")
            print(f"[GUARD] RHS str: {objRHS.original}")
        direct = test_pairwise_sentence_similarity(
            self.pairwise_similarity_cache,
            objLHS.original,
            objRHS.original,
            store=False,
            shift=False,
        )
        if _TRACE_COMPARE:
            print(f"[GUARD] direct comparison: {direct}")
            # Check if it was in cache
            cache_key = (objLHS.original, objRHS.original)
            if cache_key in self.pairwise_similarity_cache:
                print(f"[GUARD] NOTE: result was in pairwise_similarity_cache!")
        if direct == CasusHappening.INDIFFERENT:
            return CasusHappening.INDIFFERENT
        # If the full comparison didn't return INDIFFERENT but also didn't
        # return the expansion-level implication (e.g. because the relation
        # names differ and aren't linked in the ontology), fall back to a
        # property-only comparison.  The expansion already established that
        # the predicates are semantically related, so we only need to check
        # whether the properties (CAUSATION, SPACE, etc.) are compatible.
        if direct != CasusHappening.EQUIVALENT and not isImplication(direct):
            if _TRACE_COMPARE:
                print(f"[GUARD] direct was not eq/impl, checking properties directly")
            xorig = objLHS.original
            yorig = objRHS.original
            if (hasattr(xorig, 'properties') and hasattr(yorig, 'properties')
                    and xorig.properties and yorig.properties):
                xprop = set() if xorig.properties is None else xorig.properties
                yprop = set() if yorig.properties is None else yorig.properties
                dLHS = dict(xprop)
                dRHS = dict(yprop)
                keys = set(dLHS.keys()) | set(dRHS.keys())
                d = self.pairwise_similarity_cache
                for key in keys:
                    if key in dLHS and key in dRHS:
                        lhs_bests = [
                            simplifyConstituents([compare_variable(d, xx, yy) for yy in dRHS[key]])
                            for xx in dLHS[key]
                        ]
                        prop_cmp = simplifyConstituentsAcross(lhs_bests) if lhs_bests else CasusHappening.EQUIVALENT
                        if _TRACE_COMPARE:
                            print(f"[GUARD] property key={key}: prop_cmp={prop_cmp}")
                        if prop_cmp == CasusHappening.INDIFFERENT:
                            return CasusHappening.INDIFFERENT
        if _TRACE_COMPARE:
            print(f"[GUARD] returning verdict={verdict}")
        return verdict

    def searchInSet(self, lhs, rhsSet, isRightDrop = False):
        from LaSSI.HOnK.TBox.ExpandConstituents import test_pairwise_sentence_similarity, isImplication
        for rrr in rhsSet:
            rhs = rrr.bogusCopula() if isRightDrop else rrr
            val = test_pairwise_sentence_similarity(self.pairwise_similarity_cache, lhs, rhs, shift=False)
            if val == CasusHappening.EXCLUSIVES:
                return val
            if val == CasusHappening.EQUIVALENT:
                return CasusHappening.EQUIVALENT
            if isImplication(val):
                return CasusHappening.GENERAL_IMPLICATION
        return CasusHappening.INDIFFERENT

    def compare(self, objLHS:ModelSearchBasis, objRHS:ModelSearchBasis, isLeftDrop = False, isRightDrop = False)->'CasusHappening':
        cp = (objLHS.original, objRHS.original)
        if (objLHS.original == objRHS.original):
            self.main_cache[cp] = CasusHappening.EQUIVALENT
            if _TRACE_COMPARE:
                print(f"[COMPARE] path=EQUAL_ORIGINALS, result=EQUIVALENT")
            return self.main_cache[cp]
        elif ((objLHS.original == make_not(objRHS.original)) or
              (objLHS.original == make_not(objLHS.original)) or
              (make_not(objLHS.original) in objRHS.unary) or
              (make_not(objLHS.original) in objRHS.binary) or
              (make_not(objRHS.original) in objLHS.unary) or
              (make_not(objRHS.original) in objLHS.binary)):
            self.main_cache[cp] = CasusHappening.EXCLUSIVES
            if _TRACE_COMPARE:
                print(f"[COMPARE] path=NEGATION, result=EXCLUSIVES")
            return self.main_cache[cp]
        elif ((objLHS.original in objRHS.unary) or
              (objLHS.original in objRHS.binary)):
            if _TRACE_COMPARE:
                print(f"[COMPARE] path=LHS_ORIG_IN_RHS_EXPANSION, lhs_rel={getattr(objLHS.original,'rel','?')}, rhs_rel={getattr(objRHS.original,'rel','?')}")
            self.main_cache[cp] = self._guard_contextual_implication(
                objLHS, objRHS, CasusHappening.GENERAL_IMPLICATION)
            if _TRACE_COMPARE:
                print(f"[COMPARE] path=LHS_ORIG_IN_RHS_EXPANSION, result={self.main_cache[cp]}")
            return self.main_cache[cp]
        else:
            # Performing the constituents search:
            for lhs in objLHS.unary:
                negForm = make_not(lhs) if not isinstance(lhs, FNot) else lhs.arg
                if negForm in objRHS.unary:
                    self.main_cache[cp] = CasusHappening.EXCLUSIVES
                    if _TRACE_COMPARE:
                        print(f"[COMPARE] path=NEG_UNARY_EXPANSION, result=EXCLUSIVES")
                    return self.main_cache[cp]
            for lhs in objLHS.binary:
                negForm = make_not(lhs) if not isinstance(lhs, FNot) else lhs.arg
                if negForm in objRHS.binary:
                    self.main_cache[cp] = CasusHappening.EXCLUSIVES
                    if _TRACE_COMPARE:
                        print(f"[COMPARE] path=NEG_BINARY_EXPANSION, result=EXCLUSIVES")
                    return self.main_cache[cp]
            for lhs in objLHS.unary:
                if lhs in objRHS.unary:
                    if _TRACE_COMPARE:
                        print(f"[COMPARE] path=UNARY_EXPANSION_MATCH, matched={lhs}")
                    self.main_cache[cp] = self._guard_contextual_implication(
                        objLHS, objRHS, CasusHappening.GENERAL_IMPLICATION)
                    if _TRACE_COMPARE:
                        print(f"[COMPARE] path=UNARY_EXPANSION_MATCH, result={self.main_cache[cp]}")
                    return self.main_cache[cp]
            for lhs in objLHS.binary:
                if lhs in objRHS.binary:
                    if _TRACE_COMPARE:
                        print(f"[COMPARE] path=BINARY_EXPANSION_MATCH, matched={lhs}")
                    self.main_cache[cp] = self._guard_contextual_implication(
                        objLHS, objRHS, CasusHappening.GENERAL_IMPLICATION)
                    if _TRACE_COMPARE:
                        print(f"[COMPARE] path=BINARY_EXPANSION_MATCH, result={self.main_cache[cp]}")
                    return self.main_cache[cp]
            # Performing the exhaustive search:
            # Scan unary AND binary expansions for EXCLUSIVES first — a
            # genuine antonym anywhere must override any positive verdict
            # collected from the unary cross-product, otherwise a soft match
            # in unary would mask a real contradiction in binary.
            #
            # Guard against spurious EXCLUSIVES via synonym-of-mine ⇄
            # antonym-of-theirs WordNet paths: when the originals literally
            # share the same `rel` string, expansion-derived antonymy is
            # untrustworthy ("record" vs the synonym "write off" both surface
            # as antonyms in WordNet despite both being live verbs of the same
            # event).  In that case, suppress EXCLUSIVES propagation.
            same_rel_originals = (
                isinstance(objLHS.original, FBinaryPredicate)
                and isinstance(objRHS.original, FBinaryPredicate)
                and objLHS.original.rel == objRHS.original.rel
            ) or (
                isinstance(objLHS.original, FUnaryPredicate)
                and isinstance(objRHS.original, FUnaryPredicate)
                and objLHS.original.rel == objRHS.original.rel
            )
            elems = set()
            for lhs in objLHS.unary:
                if (isRightDrop) and isinstance(lhs, FNot):
                    continue
                tmp = lhs if not isLeftDrop else lhs.bogusCopula()
                val = self.searchInSet(tmp, objRHS.unary, isRightDrop)
                if val == CasusHappening.EXCLUSIVES:
                    if same_rel_originals:
                        continue
                    self.main_cache[cp] = val
                    return val
                elif val != CasusHappening.INDIFFERENT:
                    elems.add(val)
            for lhs in objLHS.binary:
                if (isRightDrop) and isinstance(lhs, FNot):
                    continue
                tmp = lhs if not isLeftDrop else lhs.bogusCopula()
                val = self.searchInSet(tmp, objRHS.binary, isRightDrop)
                if val == CasusHappening.EXCLUSIVES:
                    if same_rel_originals:
                        continue
                    self.main_cache[cp] = val
                    return val
                elif val != CasusHappening.INDIFFERENT:
                    elems.add(val)
            from LaSSI.HOnK.TBox.ExpandConstituents import simplifyConstituentsAcross
            result = simplifyConstituentsAcross(elems)
            if _TRACE_COMPARE:
                print(f"[COMPARE] path=EXHAUSTIVE_SEARCH, elems={elems}, pre_guard_result={result}")
            result = self._guard_contextual_implication(objLHS, objRHS, result)
            if _TRACE_COMPARE:
                print(f"[COMPARE] path=EXHAUSTIVE_SEARCH, post_guard_result={result}")
            self.main_cache[cp] = result
            return result
