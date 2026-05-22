import os
import json
import re
import sys
from LaSSI.structures.extended_fol.Formulae import FVariable, FNot
from LaSSI.HOnK.TBox.ComparatorUtils import isExistential

_PARA_NS = "https://ofox.co.uk/paraphrase#"
_PARA_TTL_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "Paraphrase.ttl")
)

_paraphrase_concepts_cache = None   # loaded lazily; None = not yet loaded
_asked_pairs: set = set()           # session-dedup for interactive prompt

def _load_paraphrase_ttl() -> dict:
    """Parse Paraphrase.ttl with pyoxigraph.

    Returns a dict of the form
        {concept_label: {"members": set[(form, phrase_label)],
                          "ranges":  list[(min, max)]}}

    Numeric ranges are read from `para:numericMin` / `para:numericMax` on the
    concept itself.  A concept may declare both explicit phrase members and a
    range; the range covers numbers that fall inside `[min, max]` (closed
    interval) without needing an enumerated phrase for every value.  Either
    bound may be omitted, in which case the open side is unbounded.
    """
    import pyoxigraph
    if not os.path.exists(_PARA_TTL_PATH):
        return {}
    store = pyoxigraph.Store()
    store.load(path=_PARA_TTL_PATH, format=pyoxigraph.RdfFormat.TURTLE)

    rdf_type   = pyoxigraph.NamedNode("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
    rdfs_label = pyoxigraph.NamedNode("http://www.w3.org/2000/01/rdf-schema#label")
    member_of  = pyoxigraph.NamedNode(_PARA_NS + "memberOf")
    form_pred  = pyoxigraph.NamedNode(_PARA_NS + "form")
    num_min    = pyoxigraph.NamedNode(_PARA_NS + "numericMin")
    num_max    = pyoxigraph.NamedNode(_PARA_NS + "numericMax")
    num_domain = pyoxigraph.NamedNode(_PARA_NS + "numericDomain")
    concept_cls = pyoxigraph.NamedNode(_PARA_NS + "ParaphraseConcept")
    phrase_cls  = pyoxigraph.NamedNode(_PARA_NS + "ParaphrasePhrase")

    def _to_float(literal_obj):
        try:
            return float(literal_obj.value)
        except (AttributeError, ValueError):
            return None

    concept_labels: dict = {}
    concept_ranges: dict = {}   # uri → list[(min, max)]
    concept_domains: dict = {}  # uri → set[domain_concept_label]
    for q in store.quads_for_pattern(None, rdf_type, concept_cls, None):
        uri = q.subject
        for lq in store.quads_for_pattern(uri, rdfs_label, None, None):
            concept_labels[uri.value] = lq.object.value
        mins, maxs = [], []
        for nq in store.quads_for_pattern(uri, num_min, None, None):
            v = _to_float(nq.object)
            if v is not None:
                mins.append(v)
        for nq in store.quads_for_pattern(uri, num_max, None, None):
            v = _to_float(nq.object)
            if v is not None:
                maxs.append(v)
        if mins or maxs:
            lo = min(mins) if mins else float("-inf")
            hi = max(maxs) if maxs else float("inf")
            concept_ranges.setdefault(uri.value, []).append((lo, hi))
        for dq in store.quads_for_pattern(uri, num_domain, None, None):
            # The domain object is another ParaphraseConcept; record its URI for now
            # and resolve to a label string in a second pass after `concept_labels`
            # has been fully populated.
            concept_domains.setdefault(uri.value, set()).add(dq.object.value)

    result: dict = {}
    for q in store.quads_for_pattern(None, rdf_type, phrase_cls, None):
        phrase_uri = q.subject
        label = concept_uri_str = form = None
        for lq in store.quads_for_pattern(phrase_uri, rdfs_label, None, None):
            label = lq.object.value
        for mq in store.quads_for_pattern(phrase_uri, member_of, None, None):
            concept_uri_str = mq.object.value
        for fq in store.quads_for_pattern(phrase_uri, form_pred, None, None):
            form = fq.object.value
        if label and concept_uri_str and form and concept_uri_str in concept_labels:
            concept_name = concept_labels[concept_uri_str]
            bucket = result.setdefault(concept_name, {"members": set(), "ranges": []})
            bucket["members"].add((form, label.strip().lower()))

    # Attach declared ranges to every concept that has a label, including
    # concepts that have no explicit phrase members (range-only buckets).
    for uri, ranges in concept_ranges.items():
        if uri not in concept_labels:
            continue
        concept_name = concept_labels[uri]
        bucket = result.setdefault(concept_name, {"members": set(), "ranges": [], "domains": set()})
        bucket.setdefault("domains", set())
        bucket["ranges"].extend(ranges)
    # Resolve domain URIs to concept labels.  A range-bearing concept that
    # declares `para:numericDomain para:SomeOtherConcept` only fires its range
    # when the FVariable being looked up sits inside a parent whose own
    # concept lookup yields `SomeOtherConcept` (or a parent in that set).
    for uri, domain_uris in concept_domains.items():
        if uri not in concept_labels:
            continue
        concept_name = concept_labels[uri]
        bucket = result.setdefault(concept_name, {"members": set(), "ranges": [], "domains": set()})
        bucket.setdefault("domains", set())
        for d_uri in domain_uris:
            d_label = concept_labels.get(d_uri)
            if d_label:
                bucket["domains"].add(d_label)
    # Ensure every bucket has the canonical key set even if no domain/range.
    for bucket in result.values():
        bucket.setdefault("domains", set())
        bucket.setdefault("ranges", [])
        bucket.setdefault("members", set())
    return result

def _get_paraphrase_concepts() -> dict:
    global _paraphrase_concepts_cache
    if _paraphrase_concepts_cache is None:
        _paraphrase_concepts_cache = _load_paraphrase_ttl()
    return _paraphrase_concepts_cache

def _reload_paraphrase_concepts():
    global _paraphrase_concepts_cache
    _paraphrase_concepts_cache = None

def _parse_numeric_label(nm):
    """Parse a phrase label as a number. Tolerates a trailing '%' and the
    LaSSI-specific `<dot>` placeholder that the GSM pipeline emits in place
    of '.' for entity-safe storage (e.g. `12<dot>76`).  Returns float or None.
    """
    if not isinstance(nm, str):
        return None
    s = nm.strip().lower().replace("<dot>", ".").rstrip("%").strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None

def _paraphrase_concept_of(v, parent_concept=None):
    """Resolve `v` to a Paraphrase.ttl concept label, or `None`.

    `parent_concept`, when provided, is the concept label of the FVariable
    that `v` sits inside (typically the parent of a `cop`).  It gates
    *numeric range* matches: a range-bearing concept that declares one or
    more `para:numericDomain` targets only fires for `v` when
    `parent_concept` is one of those domain labels.  Range-bearing concepts
    with no declared domain always fire regardless of `parent_concept`
    (legacy behavior); domain-declared concepts with no `parent_concept`
    skip — silence is safer than a false probability/temperature collapse.

    Explicit phrase members ("name"/"fnot_name") are always returned without
    gating: a phrase like "better-than-even" is unambiguous on its own.
    """
    concepts = _get_paraphrase_concepts()

    def _lookup(form_tag, nm):
        if not nm:
            return None
        nm = nm.strip().lower()
        # 1. Explicit phrase membership — never gated.
        for concept, bucket in concepts.items():
            if (form_tag, nm) in bucket["members"]:
                return concept
        # 2. Numeric range — only meaningful for "name" form. Gated by domain.
        if form_tag == "name":
            num = _parse_numeric_label(nm)
            if num is not None:
                for concept, bucket in concepts.items():
                    if not bucket["ranges"]:
                        continue
                    domains = bucket["domains"]
                    if domains:
                        # Range scoped to a domain — require the caller's
                        # parent_concept to be inside that domain set.
                        if parent_concept is None or parent_concept not in domains:
                            continue
                    for (lo, hi) in bucket["ranges"]:
                        if lo <= num <= hi:
                            return concept
        return None

    if isinstance(v, FVariable) and v.name:
        return _lookup("name", v.name)
    if isinstance(v, str):
        return _lookup("name", v)
    if isinstance(v, FNot):
        inner = v.arg
        if isinstance(inner, FVariable) and inner.name:
            return _lookup("fnot_name", inner.name)
        if isinstance(inner, str):
            return _lookup("fnot_name", inner)
    return None

def _paraphrase_match(lhs, rhs, lhs_parent_concept=None, rhs_parent_concept=None):
    cl = _paraphrase_concept_of(lhs, parent_concept=lhs_parent_concept)
    if cl is None:
        return False
    return cl == _paraphrase_concept_of(rhs, parent_concept=rhs_parent_concept)

def _extract_var_info(v):
    """Return (default_form, name_lower, display_str) for a variable-like expr."""
    if isinstance(v, FVariable) and v.name and not isExistential(v):
        nm = v.name.strip().lower()
        return ("name", nm, f'"{nm}"')
    if isinstance(v, FNot) and isinstance(v.arg, FVariable):
        inner = v.arg
        if inner.name and not isExistential(inner):
            nm = inner.name.strip().lower()
            return ("fnot_name", nm, f'NOT("{nm}")')
    return (None, None, None)

def _slugify(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]", "_", s.strip().lower()).strip("_")

def _to_camel(s: str) -> str:
    return "".join(w.capitalize() for w in re.split(r"[\s_]+", s.strip()))

def _append_phrase_to_ttl(concept_uri_local: str, form: str, name: str):
    slug = _slugify(name)
    lines = [
        f"\npara:phrase_{slug}\n",
        f"    a para:ParaphrasePhrase ;\n",
        f"    rdfs:label {json.dumps(name)} ;\n",
        f"    para:memberOf para:{concept_uri_local} ;\n",
        f"    para:form {json.dumps(form)} .\n",
    ]
    with open(_PARA_TTL_PATH, "a") as fh:
        fh.writelines(lines)

def _interactive_paraphrase_prompt(lhs, rhs):
    """
    If running interactively and the pair is novel, ask the user whether lhs
    and rhs should be recorded as paraphrases in Paraphrase.ttl.
    Returns True if the user confirmed and the TTL was updated.
    """
    if not sys.stdin.isatty():
        return False

    lhs_form, lhs_name, lhs_disp = _extract_var_info(lhs)
    rhs_form, rhs_name, rhs_disp = _extract_var_info(rhs)
    if lhs_name is None or rhs_name is None:
        return False

    pair_key = (lhs_disp, rhs_disp)
    if pair_key in _asked_pairs:
        return False
    _asked_pairs.add(pair_key)

    print(f"\n[Paraphrase?] {lhs_disp} vs {rhs_disp} → INDIFFERENT")
    ans = input("  Are these paraphrases of each other? [y/N]: ").strip().lower()
    if ans != "y":
        return False

    # Show existing concepts so the user can reuse one
    concepts = _get_paraphrase_concepts()
    if concepts:
        print(f"  Existing concepts: {', '.join(sorted(concepts))}")
    concept_name = input("  Concept name (existing or new): ").strip().lower()
    if not concept_name:
        print("  (skipped — no concept name given)")
        return False

    concept_uri_local = _to_camel(concept_name)
    new_concept = concept_name not in concepts

    if new_concept:
        example = input("  Example sentence (press Enter to skip): ").strip()
        lines = [
            f"\npara:{concept_uri_local}\n",
            f"    a para:ParaphraseConcept ;\n",
            f"    rdfs:label {json.dumps(concept_name)} ;\n",
            f"    rdfs:comment {json.dumps(example)} .\n",
        ]
        with open(_PARA_TTL_PATH, "a") as fh:
            fh.writelines(lines)

    # Ask/confirm the form for each side (defaulting to the natural Python type)
    lhs_form_in = input(
        f"  Form for {lhs_disp} [name/fnot_name, default: {lhs_form}]: "
    ).strip().lower()
    if lhs_form_in in ("name", "fnot_name"):
        lhs_form = lhs_form_in

    rhs_form_in = input(
        f"  Form for {rhs_disp} [name/fnot_name, default: {rhs_form}]: "
    ).strip().lower()
    if rhs_form_in in ("name", "fnot_name"):
        rhs_form = rhs_form_in

    _append_phrase_to_ttl(concept_uri_local, lhs_form, lhs_name)
    _append_phrase_to_ttl(concept_uri_local, rhs_form, rhs_name)
    _reload_paraphrase_concepts()
    print(f"  → Saved to Paraphrase.ttl under concept '{concept_name}'.")
    return True

def _canon_lookup_paraphrase_concept(name):
    """Return the Paraphrase.ttl concept name for `name` if it's an explicit
    phrase member, else None.  Used to capture the original head's concept
    for cop-magnitude gating *before* a spec-to-head swap discards it."""
    if not isinstance(name, str) or not name:
        return None
    concepts = _get_paraphrase_concepts()
    nm = name.strip().lower()
    for concept_name, bucket in concepts.items():
        if ("name", nm) in bucket.get('members', set()):
            return concept_name
    return None

def _canon_paraphrase_member(name):
    """If `name` is an explicit Paraphrase.ttl phrase, return the alphabetically-
    first phrase in the same concept (a stable canonical representative).
    Returns None if not in any concept."""
    if not isinstance(name, str) or not name:
        return None
    concepts = _get_paraphrase_concepts()
    nm = name.strip().lower()
    for _, bucket in concepts.items():
        members = bucket.get('members', set())
        if ("name", nm) in members:
            name_members = sorted({m for (form, m) in members if form == "name"})
            return name_members[0] if name_members else nm
    return None

def _canon_numeric_member(name, parent_concept):
    """If `name` parses as a number that falls in a Paraphrase.ttl numeric
    range whose declared domain matches `parent_concept`, return a canonical
    representative for that concept.  Returns None otherwise."""
    if not isinstance(name, str):
        return None
    num = _parse_numeric_label(name)
    if num is None:
        return None
    concepts = _get_paraphrase_concepts()
    for concept_name, bucket in concepts.items():
        if not bucket.get('ranges'):
            continue
        domains = bucket.get('domains') or set()
        if domains:
            if parent_concept is None or parent_concept not in domains:
                continue
        for (lo, hi) in bucket['ranges']:
            if lo <= num <= hi:
                name_members = sorted({m for (form, m) in bucket['members'] if form == "name"})
                return name_members[0] if name_members else concept_name
    return None

