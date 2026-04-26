import unittest

from LaSSI.HOnK.HOnK import HOnKSingleton, CasusHappening
from LaSSI.HOnK.TBox.ExpandConstituents import test_pairwise_sentence_similarity, compare_variable

from LaSSI.structures.extended_fol.Formulae import FUnaryPredicate, FVariable, FNot, FAnd
from extra.test_allex import ncc, ncl, all_ncc, all_ncl

traffic = FVariable('traffic', 'ENTITY')
t = FUnaryPredicate("be", traffic, 1)
tcc = FUnaryPredicate("be", traffic, 2, frozenset({"SPACE": (ncc,)}.items()))
tn = FUnaryPredicate("be", traffic, 2, frozenset({"SPACE": (ncl,)}.items()))
n_tcc = FNot(tcc)

s_3 = FAnd((t, n_tcc))


class DirectionTests(unittest.TestCase):
    def setUp(self):
        HOnKSingleton.instance()
        HOnKSingleton.init("/home/giacomo/projects/LaSSI/catabolites", "giacomo", "omocaig",
                                 "localhost", 5432, False, "/home/giacomo/projects/LaSSI/parmenides.ttl")

    def _cmpConcepts(self, x, y, case):
        val = test_pairwise_sentence_similarity({}, x, y, shift=False)
        if val != case:
            raise ValueError(f"ERROR: {val} != {case}")

    def _cmpVariables(self, x, y, case):
        val = compare_variable(dict(), x, y)
        self.assertEqual(val, case)
        if val != case:
            raise ValueError(f"ERROR: {val} != {case}")

    def test_basic(self):
        self._cmpConcepts(tn, n_tcc, CasusHappening.INDIFFERENT)
        self._cmpConcepts(tn, t, CasusHappening.GENERAL_IMPLICATION)
        self._cmpConcepts(tn, t, CasusHappening.GENERAL_IMPLICATION)
        self._cmpVariables(all_ncc, all_ncl, CasusHappening.INDIFFERENT)
        self._cmpVariables(all_ncc, ncl, CasusHappening.MISSING_1ST_IMPLICATION)
        self._cmpConcepts(tn, tcc, CasusHappening.INDIFFERENT)
        self._cmpConcepts(tcc, tn, CasusHappening.GENERAL_IMPLICATION)

    def test_last(self):
        self._cmpConcepts(tn, n_tcc, CasusHappening.INDIFFERENT)
        self._cmpConcepts(tn, t, CasusHappening.GENERAL_IMPLICATION)
        self._cmpConcepts(tn, tn, CasusHappening.EQUIVALENT)

if __name__ == "__main__":
    # print(s_3)
    # val = test_pairwise_sentence_similarity({}, t, n_tcc, shift=False)
    # print(val)
    # val = test_pairwise_sentence_similarity({}, tcc, t, shift=False)
    # print(val)
    # val = test_pairwise_sentence_similarity({}, n_tcc, t, shift=False)
    # print(val)
    # val = test_pairwise_sentence_similarity({}, t, tcc, shift=False)
    # print(val)
    HOnKSingleton.instance()
    HOnKSingleton.init("/home/giacomo/projects/LaSSI/catabolites", "giacomo", "omocaig",
                             "localhost", 5432, False, "/home/giacomo/projects/LaSSI/parmenides.ttl")

    val = test_pairwise_sentence_similarity({}, tn, n_tcc, shift=False)
    print(val)
    val = test_pairwise_sentence_similarity({}, tn, t, shift=False)
    print(val)
    val = test_pairwise_sentence_similarity({}, tn, t, shift=False)
    print(val)
    val = compare_variable({}, all_ncc, all_ncl)
    print(val)
    val = compare_variable({}, all_ncc, ncl)
    print(val)
    val = test_pairwise_sentence_similarity({}, tn, tcc, shift=False)
    print(val)
    val = test_pairwise_sentence_similarity({}, tcc, tn, shift=False)
    print(val)