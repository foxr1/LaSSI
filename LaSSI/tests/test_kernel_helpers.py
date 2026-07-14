import unittest

from LaSSI.HOnK.HOnK import CasusHappening
from LaSSI.HOnK.TBox.ConstituentComparator import _compare_single_prop_val
from LaSSI.HOnK.TBox.SpatialReasoner import _space_relation_narrowing
from LaSSI.structures.extended_fol.Formulae import FBinaryPredicate, FOr, FVariable
from LaSSI.structures.kernels.Sentence import case_in_props
from LaSSI.structures.kernels.SentenceX import get_prepositions
from LaSSI.structures.internal_graph.EntityRelationship import Singleton


class TestKernelHelpers(unittest.TestCase):
    def test_case_in_props_returns_case_values_when_requested(self):
        self.assertTrue(case_in_props({"case": "due"}))
        self.assertEqual(case_in_props({"case": "due"}, True), ["due"])
        self.assertEqual(case_in_props({"9.000000": "due"}, True), ["due"])

    def test_case_in_props_ignores_passive_and_possessive_cases(self):
        self.assertFalse(case_in_props({"case": "by"}))
        self.assertEqual(case_in_props({"case": "by"}, True), [])
        self.assertFalse(case_in_props(None))
        self.assertEqual(case_in_props(None, True), [])

    def test_sentencex_get_prepositions_reconstructs_compounds(self):
        node = Singleton(
            id=1,
            named_entity="St Nicholas' Church Yard",
            properties=frozenset({
                ("6.000000", "on"),
                ("7.000000", "or"),
                ("8.000000", "near"),
                ("11.000000", "'"),
            }),
            min=0,
            max=10,
            type="LOC",
            confidence=1.0,
        )

        self.assertIn("on or near", get_prepositions(node))

    def test_spatial_or_type_partially_matches_member(self):
        stay = FVariable("stay in place", "logical_type", None, None, None, frozenset())
        near = FVariable("near place", "logical_type", None, None, None, frozenset())
        on_or_near = FOr((stay, near))

        self.assertEqual(
            _compare_single_prop_val({}, near, on_or_near),
            CasusHappening.GENERAL_IMPLICATION,
        )
        self.assertEqual(
            _compare_single_prop_val({}, on_or_near, near),
            CasusHappening.INSTANTIATION_IMPLICATION,
        )

    def test_space_relation_narrowing_caps_on_or_near_to_near(self):
        stay = FVariable("stay in place", "logical_type", None, None, None, frozenset())
        near = FVariable("near place", "logical_type", None, None, None, frozenset())
        on_or_near_place = FVariable(
            "St Nicholas' Church Yard",
            "LOC",
            None,
            None,
            None,
            frozenset({("type", FOr((stay, near)))}),
        )
        near_place = FVariable(
            "St Nicholas' Church Yard",
            "LOC",
            None,
            None,
            None,
            frozenset({("type", "near place")}),
        )
        lhs = FBinaryPredicate("record", FVariable("?1", "existential", None, None, None, frozenset()),
                               FVariable("arson", "noun", None, None, None, frozenset()), 1.0,
                               frozenset({("SPACE", (on_or_near_place,))}))
        rhs = FBinaryPredicate("record", FVariable("?2", "existential", None, None, None, frozenset()),
                               FVariable("arson", "noun", None, None, None, frozenset()), 1.0,
                               frozenset({("SPACE", (near_place,))}))

        self.assertTrue(_space_relation_narrowing(lhs, rhs))
        self.assertFalse(_space_relation_narrowing(rhs, lhs))


if __name__ == "__main__":
    unittest.main()
