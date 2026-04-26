from LaSSI.HOnK.TBox.ExpandConstituents import compare_variable
from LaSSI.structures.extended_fol.Formulae import FVariable

ncl = FVariable("Newcastle", "GPE", None, None, 1)
all_ncl = ncl.makeAsAll()

cc = FVariable("city center", "Noun", None, None, 1)
all_cc = cc.makeAsAll()

c = FVariable("city", "Noun", None, None, 1)
all_c = c.makeAsAll()

ncc = FVariable("Newcastle", "GPE", "city center", None, 1)
all_ncc = ncc.makeAsAll()

nc = FVariable("Newcastle", "GPE", "city", None, 1)
all_nc = nc.makeAsAll()

import unittest
from LaSSI.HOnK.HOnK import HOnKSingleton, CasusHappening

class DirectionTests(unittest.TestCase):
    def setUp(self):
        HOnKSingleton.instance()
        HOnKSingleton.init("/home/giacomo/projects/LaSSI/catabolites", "giacomo", "omocaig",
                                 "localhost", 5432, False, "/home/giacomo/projects/LaSSI/parmenides.ttl")

    def _cmpVariables(self, x, y, case):
        val = compare_variable(dict(), x, y)
        self.assertEqual(val, case)
        if val != case:
            raise ValueError(f"ERROR: {val} != {case}")

    def test_all(self):
        self._cmpVariables(all_ncc, ncl, CasusHappening.MISSING_1ST_IMPLICATION)
        self._cmpVariables(all_ncc, all_ncl, CasusHappening.INDIFFERENT)
        self._cmpVariables(all_ncc, ncc, CasusHappening.EQUIVALENT)
        self._cmpVariables(all_ncc, all_ncc, CasusHappening.EQUIVALENT)
        self._cmpVariables(all_ncc, cc, CasusHappening.INSTANTIATION_IMPLICATION)
        self._cmpVariables(all_ncc, all_cc, CasusHappening.INDIFFERENT)

        self._cmpVariables(all_ncl, ncl, CasusHappening.EQUIVALENT)
        self._cmpVariables(all_ncl, all_ncl, CasusHappening.EQUIVALENT)
        self._cmpVariables(all_ncl, ncc, CasusHappening.INSTANTIATION_IMPLICATION)
        self._cmpVariables(all_ncl, all_ncc, CasusHappening.INSTANTIATION_IMPLICATION)
        self._cmpVariables(all_ncl, cc, CasusHappening.GENERAL_IMPLICATION)
        self._cmpVariables(all_ncl, all_cc, CasusHappening.INDIFFERENT)

        self._cmpVariables(all_cc, ncl, CasusHappening.GENERAL_IMPLICATION)
        self._cmpVariables(all_cc, all_ncl, CasusHappening.INDIFFERENT)
        self._cmpVariables(all_cc, ncc, CasusHappening.INSTANTIATION_IMPLICATION)
        self._cmpVariables(all_cc, all_ncc, CasusHappening.INSTANTIATION_IMPLICATION)
        self._cmpVariables(all_cc, cc, CasusHappening.EQUIVALENT)
        self._cmpVariables(all_cc, all_cc, CasusHappening.EQUIVALENT)

        ## Second shot
        self._cmpVariables(ncc, ncl, CasusHappening.MISSING_1ST_IMPLICATION)
        self._cmpVariables(ncc, all_ncl, CasusHappening.INDIFFERENT)
        self._cmpVariables(ncc, ncc, CasusHappening.EQUIVALENT)
        self._cmpVariables(ncc, all_ncc, CasusHappening.INDIFFERENT) #Done
        self._cmpVariables(ncc, cc, CasusHappening.INSTANTIATION_IMPLICATION)
        self._cmpVariables(ncc, all_cc, CasusHappening.INDIFFERENT)

        self._cmpVariables(ncl, ncl, CasusHappening.EQUIVALENT)
        self._cmpVariables(ncl, all_ncl, CasusHappening.INDIFFERENT) #Done
        self._cmpVariables(ncl, ncc, CasusHappening.INDIFFERENT)
        self._cmpVariables(ncl, all_ncc, CasusHappening.INDIFFERENT)
        self._cmpVariables(ncl, cc, CasusHappening.INDIFFERENT)
        self._cmpVariables(ncl, all_cc, CasusHappening.INDIFFERENT)

        self._cmpVariables(cc, ncl, CasusHappening.INDIFFERENT)#Done
        self._cmpVariables(cc, all_ncl, CasusHappening.INDIFFERENT) #Done
        self._cmpVariables(cc, ncc, CasusHappening.INDIFFERENT) #Done
        self._cmpVariables(cc, all_ncc, CasusHappening.INDIFFERENT) #Done
        self._cmpVariables(cc, cc, CasusHappening.EQUIVALENT)
        self._cmpVariables(cc, all_cc, CasusHappening.INDIFFERENT) #Done

        self._cmpVariables(ncc, nc, CasusHappening.GENERAL_IMPLICATION)
        self._cmpVariables(nc, ncc, CasusHappening.INDIFFERENT)


    # def test_classic_paper(self):
    #     self._cmpVariables(all_ncc, ncl, CasusHappening.INDIFFERENT)
    #     self._cmpVariables(all_ncc, ncc, CasusHappening.EQUIVALENT)
    #     self._cmpVariables(all_ncc, all_ncl, CasusHappening.INDIFFERENT)
    #     self._cmpVariables(all_ncc, all_cc, CasusHappening.INDIFFERENT)
    #     self._cmpVariables(all_ncc, ncc,CasusHappening.EQUIVALENT)
    #
    #     self._cmpVariables(all_ncl, ncl, CasusHappening.EQUIVALENT)
    #     self._cmpVariables(all_ncl, ncc, CasusHappening.INSTANTIATION_IMPLICATION)
    #     self._cmpVariables(all_ncl, all_ncc, CasusHappening.INSTANTIATION_IMPLICATION)
    #     self._cmpVariables(all_ncl, all_cc, CasusHappening.INDIFFERENT)
    #
    #     self._cmpVariables(all_cc, cc, CasusHappening.EQUIVALENT)
    #     self._cmpVariables(all_ncl,ncl, CasusHappening.EQUIVALENT)
    #
    # def test_new_all_vs_cc(self):
    #     self._cmpVariables(all_ncc, all_cc, CasusHappening.INDIFFERENT)
    #     self._cmpVariables(all_ncc, cc, CasusHappening.INSTANTIATION_IMPLICATION)
    #
    #     self._cmpVariables(all_ncl, cc, CasusHappening.GENERAL_IMPLICATION)
    #
    # def test_new_partof(self):
    #     self._cmpVariables(all_cc, ncl, CasusHappening.GENERAL_IMPLICATION)
    #     self._cmpVariables(all_cc, all_ncl, CasusHappening.INDIFFERENT)
    #     self._cmpVariables(cc, ncl, CasusHappening.INDIFFERENT)
    #     self._cmpVariables(cc, all_ncl, CasusHappening.INDIFFERENT)
    #
    #     self._cmpVariables(all_cc, ncc, CasusHappening.INSTANTIATION_IMPLICATION)
    #     self._cmpVariables(all_cc, all_ncc, CasusHappening.INSTANTIATION_IMPLICATION)
    #     self._cmpVariables(cc, ncc, CasusHappening.INDIFFERENT)
    #     self._cmpVariables(cc, all_ncc, CasusHappening.INDIFFERENT)
    #
    # def test_bare_allSomeDifference(self):
    #     self._cmpVariables( cc, all_cc,CasusHappening.INDIFFERENT)
    #     self._cmpVariables( ncl,all_ncl, CasusHappening.INDIFFERENT)
    #     self._cmpVariables( ncc, all_ncc,CasusHappening.INDIFFERENT)

    def tearDown(self):
        HOnKSingleton.stop()


if __name__ == '__main__':
    HOnKSingleton.instance()
    HOnKSingleton.init("/home/giacomo/projects/LaSSI/catabolites", "giacomo", "omocaig",
                                     "localhost", 5432, False, "/home/giacomo/projects/LaSSI/parmenides.ttl")
    print(compare_variable(dict(), nc, cc))  #Indifferent