"""Multi-token phrase loader for ontology-declared paraphrase / lifecycle phrases.

Loads the canonical multi-token phrase labels declared in
``Paraphrase.ttl`` (``ParaphrasePhrase`` instances) and
``LifecycleStates.ttl`` (``LifecyclePhrase`` instances), and exposes a
compiled regex that matches any of them in a source sentence.

This is the place to consult when you need to ask "is this surface span
an ontology-declared multi-token phrase?" — used today by the
comparison-side reasoning, and intended to be consulted by a future
**post-FOL** rule that scans an atom's source span for known phrases
and attaches them as the appropriate ``amod`` / ``cop`` / etc. property
when the upstream parse dropped them.

NOTE: an earlier attempt put this loader on the *upstream* side — a
pre-pass in ``ResolveSingleSentence`` that emitted synthetic
``MeuDBEntry`` records for each match so ``TypeResolver.mergeMeuNodes``
would fold them into a single node.  That caused structural regressions
downstream: preposition-internal tokens like ``than`` (in
``better-than-even``) and ``in`` (in ``one-in-four``) carried dependency
edges that, after merging, surfaced as spurious properties on the wrong
host (``COMPARISON:better-than-even`` on the kernel root,
``amod:sunny`` on a merged ``sunny spells``).  Picking a "syntactic
head" of an arbitrary multi-token phrase is genuinely ambiguous — the
head is right-most in ``rain shower`` / ``sunny spells`` / ``one-in-four``
but left-most in comparative ADJPs like ``better-than-even`` — so no
single rule covers all cases.

The right place to recover lost phrases is therefore *after* the FOL is
built, where each atom's source-text span is known and the recovery can
attach the phrase to the right host without disturbing the dependency
graph.  This loader stays in place so that recovery rule has a single
source of truth shared with the comparator's Paraphrase/Lifecycle
lookups.
"""

import os
import re


_PARA_TTL_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "Paraphrase.ttl")
)
_LIFECYCLE_TTL_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "LifecycleStates.ttl")
)

_PARA_NS = "https://ofox.co.uk/paraphrase#"
_LIFECYCLE_NS = "https://ofox.co.uk/lifecycle#"

# Cached compiled regex + raw phrase set, both invalidated when either TTL's
# mtime advances past the cached value.
_cached_regex = None
_cached_phrases: set = set()
_cached_mtimes: tuple = (None, None)


def _ttl_phrase_labels(ttl_path, ns, phrase_class_local):
    """Return the lowercase rdfs:label of every ParaphrasePhrase /
    LifecyclePhrase declared in the TTL.  Returns an empty set if pyoxigraph
    isn't available or the file is missing — the caller treats the recovery
    as a best-effort augmentation."""
    if not os.path.exists(ttl_path):
        return set()
    try:
        import pyoxigraph
    except ImportError:
        return set()
    store = pyoxigraph.Store()
    store.load(path=ttl_path, format=pyoxigraph.RdfFormat.TURTLE)
    rdf_type = pyoxigraph.NamedNode("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
    rdfs_label = pyoxigraph.NamedNode("http://www.w3.org/2000/01/rdf-schema#label")
    phrase_cls = pyoxigraph.NamedNode(ns + phrase_class_local)
    labels = set()
    for q in store.quads_for_pattern(None, rdf_type, phrase_cls, None):
        for lq in store.quads_for_pattern(q.subject, rdfs_label, None, None):
            v = lq.object.value
            if isinstance(v, str):
                labels.add(v.strip().lower())
    return labels


def _is_multi_token(phrase):
    """A phrase needs upstream-tokeniser recovery only when it contains a
    hyphen or whitespace — single tokens come through Stanza intact."""
    return bool(phrase) and (any(c.isspace() for c in phrase) or "-" in phrase)


def _load_all_multi_token_phrases():
    para = _ttl_phrase_labels(_PARA_TTL_PATH, _PARA_NS, "ParaphrasePhrase")
    life = _ttl_phrase_labels(_LIFECYCLE_TTL_PATH, _LIFECYCLE_NS, "LifecyclePhrase")
    return {p for p in (para | life) if _is_multi_token(p)}


def _build_regex(phrases):
    if not phrases:
        return None
    # Longest first so "one-in-four" beats a hypothetical "one" entry, and
    # "clear skies" beats "clear".  Word-boundary anchors avoid matching
    # inside larger words.
    parts = sorted(phrases, key=len, reverse=True)
    pattern = r"\b(?:" + "|".join(re.escape(p) for p in parts) + r")\b"
    return re.compile(pattern, re.IGNORECASE)


def _current_mtimes():
    def _mtime(path):
        try:
            return os.path.getmtime(path)
        except OSError:
            return None
    return (_mtime(_PARA_TTL_PATH), _mtime(_LIFECYCLE_TTL_PATH))


def get_multi_token_phrase_regex():
    """Return a compiled regex matching any multi-token phrase declared in
    Paraphrase.ttl or LifecycleStates.ttl, or `None` if none are declared.

    The regex is case-insensitive and word-boundary anchored.  Results are
    cached and invalidated whenever either TTL's mtime advances.
    """
    global _cached_regex, _cached_phrases, _cached_mtimes
    mt = _current_mtimes()
    if _cached_regex is None or mt != _cached_mtimes:
        _cached_phrases = _load_all_multi_token_phrases()
        _cached_regex = _build_regex(_cached_phrases)
        _cached_mtimes = mt
    return _cached_regex


def get_multi_token_phrase_set():
    """Return the set of multi-token phrase labels driving the regex above."""
    get_multi_token_phrase_regex()  # populate cache
    return set(_cached_phrases)
