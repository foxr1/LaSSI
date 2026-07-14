"""Unit tests for the pre-CoreNLP chunking layer.

These exercise the structural splitter (:mod:`LaSSI.phases.StructuredSentenceLoader`),
the role profiler (:mod:`LaSSI.ner.ChunkProfiler`), and the chunk-flow helpers
on :class:`LaSSI.phases.RowChunkPipeline.RowChunkPipeline` in isolation. A
lightweight mock HOnK loads the real ``raw_data`` vocabulary files so the suite
runs without the full ontology graph store.
"""

import os
import unittest

from LaSSI.ner.ChunkProfiler import ChunkRole, profile
from LaSSI.phases.RowChunkPipeline import RowChunkPipeline
from LaSSI.phases.StructuredSentenceLoader import (
    StructuredChunk,
    split_structured,
    split_structured_texts,
)

_RAW = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "raw_data")
)


def _load(rel):
    path = os.path.join(_RAW, rel)
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return {line.strip().lower() for line in f if line.strip()}


class MockHOnK:
    """Loads the real raw_data lookup sets used by the chunker."""

    def getMaterialisationVerbs(self):
        return _load("verbs/materialisation_verbs.txt")

    def getCausativeVerbs(self):
        return _load("verbs/causative_verbs.txt")

    def getMovementVerbs(self):
        return _load("verbs/movement_verbs.txt")

    def getMeansVerbs(self):
        return _load("verbs/means_verbs.txt")

    def getPredictionVerbs(self):
        return _load("verbs/prediction_verbs.txt")

    def getStateVerbs(self):
        return _load("verbs/state_verbs.txt")

    def getPhrasalVerbs(self):
        return _load("verbs/phrasal_verbs.txt")

    def getTransitiveVerbs(self):
        return _load("verbs/transitive_verbs.txt")

    def getModalAdjectives(self):
        return _load("adjectives/modal_adjectives.txt")

    def getStatusNouns(self):
        return _load("nouns/status_nouns.txt")

    def getServiceStateNouns(self):
        return _load("nouns/service_state_nouns.txt")

    def getWeatherConditionNouns(self):
        return _load("nouns/weather_condition_nouns.txt")

    def getFieldLabelNouns(self):
        return _load("nouns/field_label_nouns.txt")

    def getCopulaSurfaceForms(self):
        return _load("verbs/copula_surface_forms.txt") or {
            "is", "are", "was", "were", "be", "been", "being", "am",
        }

    def getPrepositions(self):
        return set()


def _profile(text, honk, is_label=False):
    return profile(StructuredChunk(text=text, is_label=is_label), honk)


class TestStructuredSplitter(unittest.TestCase):
    def test_inch_mark_is_protected(self):
        # 6" must not be split off as a delimiter; whole NP stays in one chunk.
        chunks = split_structured_texts('Abandon approx 11m of 6" SI')
        self.assertEqual(chunks, ['Abandon approx 11m of 6" SI'])

    def test_slash_date_range_is_protected(self):
        chunks = split_structured_texts("Work dates: 13/04/2026 - 17/04/2026")
        self.assertIn("13/04/2026 - 17/04/2026", chunks[-1])

    def test_colon_marks_label(self):
        chunks = split_structured("Traffic management: Some carriageway incursion.")
        self.assertTrue(chunks[0].is_label)
        self.assertEqual(chunks[1].delim_before, "COLON")

    def test_terminal_period_kept_on_chunk(self):
        chunks = split_structured_texts("First thing. Second thing.")
        self.assertTrue(chunks[0].endswith("."))


class TestChunkProfiler(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.honk = MockHOnK()

    def test_modal_aux_downgrades_to_prose(self):
        # "Work expected to end <date>" must NOT be ACTION or TIME_RANGE —
        # the modal adjective "expected" marks a passive/raising clause.
        role = _profile("Work expected to end 2026-04-27", self.honk)
        self.assertEqual(role, ChunkRole.PROSE)

    def test_imperative_action_still_classified(self):
        self.assertEqual(
            _profile('Abandon approx 11m of 6" SI', self.honk),
            ChunkRole.ACTION,
        )

    def test_bare_date_range_is_time_range(self):
        self.assertEqual(
            _profile("13/04/2026 - 17/04/2026", self.honk),
            ChunkRole.TIME_RANGE,
        )

    def test_field_label_is_header(self):
        self.assertEqual(
            _profile("Traffic management", self.honk, is_label=True),
            ChunkRole.HEADER,
        )


class TestRowChunkPipelineHelpers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rcp = RowChunkPipeline(honk=MockHOnK(), services=None)

    def test_extract_np_recovers_compound_modifier(self):
        head, mods = self.rcp._extract_chunk_np(
            "also a connection permit for works on March Road..", "also",
        )
        self.assertEqual(head, "permit")
        self.assertEqual(mods, ["connection"])

    def test_elided_copula_injected_for_telegraphic_periphrasis(self):
        # "Work expected to end <date>" -> "Work is expected to end <date>"
        self.assertEqual(
            self.rcp._inject_elided_copula("Work expected to end 2026-04-27."),
            "Work is expected to end 2026-04-27.",
        )

    def test_elided_copula_plural_subject_uses_are(self):
        self.assertEqual(
            self.rcp._inject_elided_copula("Lane closures scheduled to begin Monday"),
            "Lane closures are scheduled to begin Monday",
        )

    def test_no_copula_injection_without_to(self):
        # "Delays unlikely" is a bare modal predication, not a periphrasis.
        self.assertEqual(
            self.rcp._inject_elided_copula("Delays unlikely."),
            "Delays unlikely.",
        )

    def test_no_copula_injection_when_already_present(self):
        self.assertEqual(
            self.rcp._inject_elided_copula("Work is expected to end 2026-04-27."),
            "Work is expected to end 2026-04-27.",
        )

    def test_no_copula_injection_in_reduced_relative_with_copula(self):
        # "with parakeets expected to fly over" is a reduced relative inside a
        # full clause ("Newcastle is forecast to ..."); the existing copula
        # must suppress injection so the sentence isn't corrupted.
        s = ("Newcastle is forecast to have a sunny day at 11am on 13 April "
             "2026 with parakeets expected to fly over.")
        self.assertEqual(self.rcp._inject_elided_copula(s), s)

    def test_bare_np_body_is_not_rejoined(self):
        # A bare-NP body parses fine standalone; rejoining behind a colon
        # makes the kernel builder drop it, so it must stay separate.
        chunks = [
            StructuredChunk(text="Traffic management", is_label=True,
                            delim_before="ROW_START"),
            StructuredChunk(text="Some carriageway incursion", is_label=False,
                            delim_before="COLON"),
        ]
        out = self.rcp._rejoin_label_with_verbless_body(chunks)
        self.assertEqual(len(out), 2)


if __name__ == "__main__":
    unittest.main()
