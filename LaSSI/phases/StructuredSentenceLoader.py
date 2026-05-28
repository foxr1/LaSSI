"""Structured pre-parser that splits scraped notice rows into semantically
coherent chunks before CoreNLP sees them.

The default :func:`split_yaml_row_sentences` in
:mod:`LaSSI.phases.SentenceLoader` only splits on ``.!?`` followed by an
uppercase letter, so notice formats like ``Outcome: ...``,
``Traffic management: ...``, ``13/04/2026 - 17/04/2026``, hyphen-glued date
pairs, and metric lists are forced into a single dependency tree. This module
splits structurally while *protecting* spans that look like delimiters but
are actually atomic (ISO timestamps, decimals with units, percentages, slash
dates, slash names, hyphenated lexical phrases).

Role classification lives in :mod:`LaSSI.ner.ChunkProfiler` — this module is
intentionally vocabulary-free.
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class StructuredChunk:
    """One structural chunk of a row.

    ``is_label`` is True iff the chunk is immediately followed by a ``:`` —
    i.e. it is the header label of a ``Header: body`` pair. Downstream
    profiling reads this together with the chunk's own head noun.

    ``delim_before`` names the delimiter that *introduced* this chunk
    (``ROW_START``, ``PERIOD``, ``COLON``, ``SEMICOLON``, ``HYPHEN``). It is
    informational for the profiler and the row-merger; the splitter itself
    does not assign roles.
    """
    text: str
    is_label: bool = False
    delim_before: str = "ROW_START"


# --- Protected spans -------------------------------------------------------
# Each pattern matches a contiguous substring that must survive splitting
# intact. Order matters: more specific patterns first so they win the regex
# alternation in :data:`_PROTECTED_RE`.
_PROTECTED_PATTERNS: Tuple[str, ...] = (
    # ISO timestamps: 2026-04-14T14:00Z, 2026-04-14T14:00:00+00:00
    r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2})?(?:Z|[+\-]\d{2}:?\d{2})?',
    # Hyphen-glued weekday-date pair without a space before the hyphen:
    # "Monday, 13 April 2026- Friday 17 April"
    r'(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)[a-z]*,?\s+\d{1,2}\s+[A-Z][a-z]+'
    r'(?:\s+\d{4})?-\s*(?:(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)[a-z]*,?\s+)?'
    r'\d{1,2}\s+[A-Z][a-z]+(?:\s+\d{4})?',
    # Slash-date ranges and slash dates: 13/04/2026 - 17/04/2026, 13/04/2026
    r'\d{1,4}/\d{1,2}/\d{1,4}\s*-\s*\d{1,4}/\d{1,2}/\d{1,4}',
    r'\d{1,4}/\d{1,2}/\d{1,4}',
    # Time windows and bare times: 08:00 - 16:00, 14:00
    r'\d{1,2}:\d{2}\s*-\s*\d{1,2}:\d{2}',
    r'\d{1,2}:\d{2}',
    # Number + unit (temperatures, speeds, lengths): 12.76°C, 3.09 mph
    r'\d+(?:\.\d+)?\s*(?:°[CF]|mph|km/h|kph|kg|km|cm|mm|in\b|ft\b|m\b)',
    # Percentages: 57%, 12.5%
    r'\d+(?:\.\d+)?\s*%',
    # Inch / foot prime / double-prime marks: 6", 11'
    r"\d+(?:\.\d+)?\s*['\"′″]",
    # Bare decimals (last among numeric patterns so units win)
    r'\d+\.\d+',
    # Slash names: Bus/Coach, North/South
    r'[A-Z][a-z]+/[A-Z][a-z]+',
    # Hyphenated lexical phrases: better-than-even, all-day
    r'[a-zA-Z]+(?:-[a-zA-Z]+){1,}',
)

_PROTECTED_RE = re.compile('|'.join(f'(?:{p})' for p in _PROTECTED_PATTERNS))

# Structural delimiters
# Captured groups name the delimiter so the splitter can record what
# introduced each chunk.
#
# The period boundary is intentionally zero-width on the period side: the
# match consumes only the whitespace between a `.` and the next uppercase
# letter, leaving the period attached to the previous chunk. CoreNLP uses
# the terminal `.` as a subject-attachment anchor, so dropping it (as the
# previous consuming form `\.+\s+` did) caused chunks like "Bicycle theft
# recorded on or near Edward Place ... in January 2026" to lose their
# subject in the parse.
_DELIM_RE = re.compile(
    r'(?P<period>(?<=\.)\s+(?=[A-Z]))'
    r'|(?P<colon>:\s+(?=\S))'
    r'|(?P<semicolon>;\s*)'
    r'|(?P<hyphen>\s+-\s+)'
)

_TRAILING_WS_RE = re.compile(r'\s+$')


def _find_protected_spans(text: str) -> List[Tuple[int, int]]:
    return [(m.start(), m.end()) for m in _PROTECTED_RE.finditer(text)]


def _pos_in_spans(pos: int, spans: List[Tuple[int, int]]) -> bool:
    for s, e in spans:
        if s <= pos < e:
            return True
    return False


def split_structured(text: Optional[str]) -> List[StructuredChunk]:
    """Split ``text`` into structural chunks while protecting atomic spans.

    Returns an empty list for empty input. A row that contains no recognised
    structural delimiter is returned as a single chunk so callers can treat
    every row uniformly.
    """
    if not isinstance(text, str) or not text.strip():
        return []
    text = text.strip()
    spans = _find_protected_spans(text)

    chunks: List[StructuredChunk] = []
    last_pos = 0
    delim_before = "ROW_START"

    for m in _DELIM_RE.finditer(text):
        if _pos_in_spans(m.start(), spans):
            continue
        delim_name = m.lastgroup.upper()
        # The period split is zero-width on the period side, so an
        # intermediate chunk introduced by a `.` boundary keeps its
        # terminal period(s) — CoreNLP needs them as a subject anchor.
        # We only strip surrounding whitespace.
        seg = text[last_pos:m.start()].strip()
        if seg:
            chunks.append(StructuredChunk(
                text=seg,
                is_label=(delim_name == "COLON"),
                delim_before=delim_before,
            ))
        delim_before = delim_name
        last_pos = m.end()

    # Final tail: preserve a natural sentence-terminating period — CoreNLP
    # uses it as an anchor for subject attachment.
    tail = _TRAILING_WS_RE.sub('', text[last_pos:])
    if tail:
        chunks.append(StructuredChunk(
            text=tail,
            is_label=False,
            delim_before=delim_before,
        ))

    return chunks if chunks else [StructuredChunk(text=text)]


def split_structured_texts(text: Optional[str]) -> List[str]:
    """Convenience wrapper returning just the chunk strings, for callers that
    don't need delimiter metadata. Mirrors the signature of
    :func:`LaSSI.phases.SentenceLoader.split_yaml_row_sentences`."""
    return [c.text for c in split_structured(text)]
