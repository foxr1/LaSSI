"""Role classifier for structural chunks produced by
:mod:`LaSSI.phases.StructuredSentenceLoader`.

Roles drive downstream behaviour:

* :func:`LaSSI.LaSSI.LaSSI._merge_intermediate_per_row` picks the primary
  kernel by role priority (``ACTION`` > substantive ``PROSE`` >
  ``REPORT_HEADER`` > ``HEADER``) and promotes construct-key properties from
  the rest onto it.
* :mod:`LaSSI.ner.TypeResolver` reads the role of an ``ACTION`` / ``STATUS``
  chunk so that an offence-style head noun is not silently resolved as a
  location.
"""

import re
from enum import Enum
from typing import Iterable, Optional

from LaSSI.phases.StructuredSentenceLoader import StructuredChunk


class ChunkRole(str, Enum):
    ACTION = "ACTION"
    PROSE = "PROSE"
    HEADER = "HEADER"
    REPORT_HEADER = "REPORT_HEADER"
    ATTRIBUTE = "ATTRIBUTE"
    STATUS = "STATUS"
    TIME_RANGE = "TIME_RANGE"
    CONTEXT = "CONTEXT"


_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*")
_DATE_LIKE_RE = re.compile(r'\d{1,4}[/\-]\d{1,2}[/\-]\d{1,4}|\d{1,2}:\d{2}|\d{4}-\d{2}-\d{2}')
_METRIC_MARKER_RE = re.compile(r'\d+(?:\.\d+)?\s*(?:°[CF]|%|mph|km/h|kph|kg|km|cm|mm)')

# Leading particles that are not the head of a chunk.
_LEADING_SKIP = {
    "the", "a", "an", "to", "for", "of", "on", "in", "at", "by",
    "with", "from", "as", "into", "onto",
    "no", "not", "never",
}


def _tokens(text: str) -> list:
    return [m.lower() for m in _TOKEN_RE.findall(text)]


def _get_set(honk, accessor: str) -> set:
    if honk is None:
        return set()
    fn = getattr(honk, accessor, None)
    if fn is None:
        return set()
    try:
        return fn() or set()
    except Exception:
        return set()


def _any_in(tokens: Iterable[str], lookup: set) -> bool:
    if not lookup:
        return False
    return any(t in lookup for t in tokens)


def _phrase_in(phrase: str, lookup: set) -> bool:
    return bool(lookup) and phrase.lower() in lookup


# Verb classes consulted for ACTION classification. Deliberately narrow:
# HOnK fuzzy-loads many common nouns and particles under TransitiveVerb
# (e.g. "bicycle", "on", "no") so including that class would make the
# profiler treat ordinary subject-led prose like "Bicycle theft recorded
# ..." as an imperative ACTION. The imperative roadworks/notice
# vocabulary (abandon, demolish, install, rebuild, replace, excavate,
# renew) lives in CausativeVerb / MaterialisationVerb, and forecast/
# predict/report — the verbs that anchor REPORT_HEADER chunks — live in
# PredictionVerb. State and movement verbs cover the remaining tightly-
# imperative cases without admitting nominal fuzzy-matches.
_ACTION_VERB_ACCESSORS = (
    "getCausativeVerbs",
    "getMaterialisationVerbs",
    "getMovementVerbs",
    "getMeansVerbs",
    "getPredictionVerbs",
)

# Wider verb set used for the STATUS-check's "head is a real verb" gate.
# A chunk that mentions a status_noun but is led by a known transitive
# verb is usually a sentence about a status, not a status fragment, so
# we want the broader check there.
_VERB_ACCESSORS = _ACTION_VERB_ACCESSORS + (
    "getStateVerbs",
    "getPhrasalVerbs",
    "getTransitiveVerbs",
)


def _is_known_verb(honk, term: str) -> bool:
    if not term:
        return False
    for acc in _VERB_ACCESSORS:
        if term in _get_set(honk, acc):
            return True
    return False


def _is_imperative_verb(honk, term: str) -> bool:
    if not term:
        return False
    for acc in _ACTION_VERB_ACCESSORS:
        if term in _get_set(honk, acc):
            return True
    return False


def _has_copula(honk, tokens: Iterable[str]) -> bool:
    copulas = _get_set(honk, "getCopulaSurfaceForms")
    if not copulas:
        copulas = {"is", "are", "was", "were", "be", "been", "being", "am"}
    return any(t in copulas for t in tokens)


def profile(chunk: StructuredChunk, honk) -> ChunkRole:
    """Assign a :class:`ChunkRole` to ``chunk``. Conservative: anything not
    confidently structural defaults to :attr:`ChunkRole.PROSE` so that
    CoreNLP still gets a chance to parse it."""
    text = (chunk.text or "").strip()
    if not text:
        return ChunkRole.CONTEXT

    tokens = _tokens(text)
    lowered = text.lower()

    field_labels = _get_set(honk, "getFieldLabelNouns")
    status_nouns = _get_set(honk, "getStatusNouns")
    service_state_nouns = _get_set(honk, "getServiceStateNouns")
    weather_nouns = _get_set(honk, "getWeatherConditionNouns")
    prediction_verbs = _get_set(honk, "getPredictionVerbs")

    # Label chunks (followed by ':')
    if chunk.is_label:
        if _any_in(tokens, weather_nouns) and _any_in(tokens, prediction_verbs):
            return ChunkRole.REPORT_HEADER
        if (_any_in(tokens, field_labels) or _phrase_in(lowered, field_labels)
                or _any_in(tokens, status_nouns)
                or _any_in(tokens, service_state_nouns)):
            return ChunkRole.HEADER
        # The colon is itself a structural cue: the chunk before it is a
        # label by construction, even if no vocabulary class matched.
        return ChunkRole.HEADER

    # Non-label chunks
    # TIME_RANGE: dominated by date/time literals, no verb
    if _DATE_LIKE_RE.search(text) and not any(_is_known_verb(honk, t) for t in tokens):
        return ChunkRole.TIME_RANGE

    # ATTRIBUTE: numeric value adjacent to a unit/percent. We only count
    # number+unit composites — short UnitOfMeasure tokens (`in`, `m`,
    # `hr`, `g`) collide with common English words, so token-level
    # membership is unsafe. A copula nearby downgrades to PROSE
    # ("Temperature is 12°C" is a sentence about the metric, not a
    # bare metric chunk).
    if _METRIC_MARKER_RE.search(text) and not _has_copula(honk, tokens):
        return ChunkRole.ATTRIBUTE

    # Walk past leading particles (articles, prepositions, negators) to
    # find the effective clause head. Used by both STATUS and ACTION
    # checks below.
    head_idx = 0
    while head_idx < len(tokens) and tokens[head_idx] in _LEADING_SKIP:
        head_idx += 1

    # STATUS: short nominal/participial chunk that mentions a status
    # noun. "Awaiting court outcome", "Investigation complete",
    # "no suspect identified" all state a status, not an event — even
    # when one of the surface tokens is technically a transitive verb
    # in HOnK ("suspect", "complete"). The brevity is the real signal:
    # status fragments don't carry full SVO clauses, they're nominal.
    if _any_in(tokens, status_nouns):
        effective_tokens = tokens[head_idx:]
        if len(effective_tokens) <= 4:
            return ChunkRole.STATUS
        # Longer chunks: only treat as STATUS if the head itself isn't a
        # real verb (preserves the original signal for borderline cases).
        if head_idx >= len(tokens) or not _is_known_verb(honk, tokens[head_idx]):
            return ChunkRole.STATUS

    # ACTION: clause-initial imperative verb (after the same skip set),
    # with no copula in the chunk. Use the narrow imperative-verb set —
    # transitive-verb membership alone is too noisy (HOnK fuzzy-loads
    # nouns like "bicycle" / particles like "on" into TransitiveVerb).
    if head_idx < len(tokens):
        head = tokens[head_idx]
        if _is_imperative_verb(honk, head) and not _has_copula(honk, tokens):
            return ChunkRole.ACTION

    return ChunkRole.PROSE
