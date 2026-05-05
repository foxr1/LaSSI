# from LaSSI.HOnK.TBox.ExpandConstituents import CasusHappening, test_pairwise_sentence_similarity, isImplication
# from logical_repr.Sentences import FUnaryPredicate, FBinaryPredicate, FNot
# from logical_repr.rewrite_kernels import make_not
from pydatagramdb import result

from LaSSI.structures.extended_fol.Formulae import *
from LaSSI.HOnK.HOnK import CasusHappening


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
            return self.main_cache[cp]
        elif ((objLHS.original == make_not(objRHS.original)) or
              (objLHS.original == make_not(objLHS.original)) or
              (make_not(objLHS.original) in objRHS.unary) or
              (make_not(objLHS.original) in objRHS.binary) or
              (make_not(objRHS.original) in objLHS.unary) or
              (make_not(objRHS.original) in objLHS.binary)):
            self.main_cache[cp] = CasusHappening.EXCLUSIVES
            return self.main_cache[cp]
        elif ((objLHS.original in objRHS.unary) or
              (objLHS.original in objRHS.binary)):
            self.main_cache[cp] = CasusHappening.GENERAL_IMPLICATION
            return self.main_cache[cp]
        else:
            # Performing the constituents search:
            for lhs in objLHS.unary:
                negForm = make_not(lhs) if not isinstance(lhs, FNot) else lhs.arg
                if negForm in objRHS.unary:
                    self.main_cache[cp] = CasusHappening.EXCLUSIVES
                    return self.main_cache[cp]
            for lhs in objLHS.binary:
                negForm = make_not(lhs) if not isinstance(lhs, FNot) else lhs.arg
                if negForm in objRHS.binary:
                    self.main_cache[cp] = CasusHappening.EXCLUSIVES
                    return self.main_cache[cp]
            for lhs in objLHS.unary:
                if lhs in objRHS.unary:
                    self.main_cache[cp] = CasusHappening.GENERAL_IMPLICATION
                    return self.main_cache[cp]
            for lhs in objLHS.binary:
                if lhs in objRHS.binary:
                    self.main_cache[cp] = CasusHappening.GENERAL_IMPLICATION
                    return self.main_cache[cp]
            # Performing the exhaustive search:
            # Scan unary AND binary expansions for EXCLUSIVES first — a
            # genuine antonym anywhere must override any positive verdict
            # collected from the unary cross-product, otherwise a soft match
            # in unary would mask a real contradiction in binary.
            elems = set()
            for lhs in objLHS.unary:
                if (isRightDrop) and isinstance(lhs, FNot):
                    continue
                tmp = lhs if not isLeftDrop else lhs.bogusCopula()
                val = self.searchInSet(tmp, objRHS.unary, isRightDrop)
                if val == CasusHappening.EXCLUSIVES:
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
                    self.main_cache[cp] = val
                    return val
                elif val != CasusHappening.INDIFFERENT:
                    elems.add(val)
            from LaSSI.HOnK.TBox.ExpandConstituents import simplifyConstituentsAcross
            result = simplifyConstituentsAcross(elems)
            self.main_cache[cp] = result
            return result
