"""Temporal-contradiction detection for the eFOL comparison.

Mirrors :mod:`LaSSI.HOnK.TBox.SpatialReasoner`'s ``_space_mismatch_contradiction``
but for *defined* calendar dates. Two predicates that each assert a defined
date which differ describe mutually exclusive timings of the same event and
cannot both hold — e.g. roadworks "expected to end 2026-04-27" vs
"expected to end on 30 April".

Deliberate semantics (see plan / ModelSearch SPACE-TIME note):
  * Fires ONLY when BOTH sides carry at least one defined date AND the date
    sets are disjoint. A defined date vs **no** date stays indifferent (this
    protects weather forecasts, where a date compared against a dateless
    sentence is extra detail, not a contradiction).
  * "Defined" = a concrete ``YYYY-MM-DD`` calendar day. Relative/continuous
    times ("before", "during routine monitoring") carry no such day and are
    ignored — so the rule is conservative.

Dates are searched recursively through the formula, including nested
``SENTENCE`` clauses and predicate arguments, because the same calendar fact
can surface either as a top-level ``TIME`` property (S2) or buried inside a
preserved sub-clause (S0's ``… end(…, 2026-04-27)``).

Written so it can later be extended to date *ranges* (cf. roadworks_003):
``_defined_dates`` would additionally yield (start, end) interval tokens and
``_time_mismatch_contradiction`` would test interval disjointness.
"""

import re

from LaSSI.structures.extended_fol.Formulae import (
    FVariable,
    FUnaryPredicate,
    FBinaryPredicate,
    FAnd,
    FOr,
    FNot,
)

_DATE_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
_DATE_TYPES = {"DATE", "TIME", "SUTIME"}
_MAX_DEPTH = 8


def _normalise_day(value) -> str:
    """Return the ``YYYY-MM-DD`` day for a date-like value, or None."""
    if value is None:
        return None
    m = _DATE_RE.search(str(value))
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else None


def _defined_dates(node, depth: int = 0) -> set:
    """Collect every defined ``YYYY-MM-DD`` day reachable in ``node`` —
    its TIME-typed variables, their properties, predicate args, and nested
    SENTENCE clauses."""
    out = set()
    if node is None or depth > _MAX_DEPTH:
        return out

    if isinstance(node, FVariable):
        if (getattr(node, "type", "") or "").upper() in _DATE_TYPES:
            day = _normalise_day(getattr(node, "name", None))
            if day:
                out.add(day)
        for _, vs in (getattr(node, "properties", None) or ()):
            for v in (vs if isinstance(vs, (tuple, list)) else (vs,)):
                out |= _defined_dates(v, depth + 1)
        return out

    if isinstance(node, (FUnaryPredicate, FBinaryPredicate)):
        for _, vs in (getattr(node, "properties", None) or ()):
            for v in (vs if isinstance(vs, (tuple, list)) else (vs,)):
                out |= _defined_dates(v, depth + 1)
        for attr in ("arg", "src", "dst"):
            out |= _defined_dates(getattr(node, attr, None), depth + 1)
        return out

    if isinstance(node, (FAnd, FOr)):
        for a in (getattr(node, "args", None) or ()):
            out |= _defined_dates(a, depth + 1)
        return out

    if isinstance(node, FNot):
        out |= _defined_dates(getattr(node, "arg", None), depth + 1)
        return out

    return out


def _time_mismatch_contradiction(lhs, rhs) -> bool:
    """True iff LHS and RHS each assert at least one defined date and the
    two date sets are disjoint (no shared day)."""
    lhs_days = _defined_dates(lhs)
    rhs_days = _defined_dates(rhs)
    if not lhs_days or not rhs_days:
        return False
    return not (lhs_days & rhs_days)
