import re

_DATETIME_RE = re.compile(
    r"""^
        (?P<y>\d{4})
        [-/](?P<m>\d{1,2})
        [-/](?P<d>\d{1,2})
        (?:                              # optional time component
          [tT \-]                        #   T/space/dash separator
          (?P<H>\d{1,2})
          [:\-]?(?P<M>\d{1,2})?          #   minutes (optional)
          (?:[:\-](?P<S>\d{1,2}))?       #   seconds (optional)
          (?:\.\d+)?                     #   fractional seconds (discarded)
          (?:[zZ]|[+\-]\d{2}:?\d{2})?    #   timezone marker (Z or ±HHMM)
        )?
        $""",
    re.VERBOSE,
)


def canonicalize_datetime_string(s):
    """Return the canonical `YYYY-MM-DDTHH:MM:SSZ` form of *s*, or `None`."""
    if not isinstance(s, str):
        return None
    s2 = s.strip().replace("<dot>", ".")
    if not s2:
        return None
    m = _DATETIME_RE.match(s2)
    if m is None:
        return None
    try:
        y = int(m.group("y"))
        mo = int(m.group("m"))
        d = int(m.group("d"))
        h = int(m.group("H") or 0)
        mi = int(m.group("M") or 0)
        se = int(m.group("S") or 0)
    except (TypeError, ValueError):
        return None
    if not (1 <= mo <= 12 and 1 <= d <= 31 and 0 <= h <= 23
            and 0 <= mi <= 59 and 0 <= se <= 59):
        return None
    return f"{y:04d}-{mo:02d}-{d:02d}T{h:02d}:{mi:02d}:{se:02d}Z"
