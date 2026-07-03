"""Ontology-grounded LLM entailment backend.

This mirrors `LaSSI.similarities.LLM.LLMPrompt` (same Ollama client, JSON
output schema and error handling, reached via subclassing) but **grounds** the
model's implication judgment in facts drawn from LaSSI's symbolic knowledge,
following the "Follow the Path" recipe (arXiv:2505.11140): extract the salient
terms/phrases of the two sentences, look up the verified relations between them,
and inject those relations into the prompt as evidence.

Three knowledge sources can be combined, each ablatable (see `sources`):
  * "honk"       — the general HOnK ontology via `HOnK.name_eq`
                   (synonymy / isA / partOf / sparse neqTo).
  * "lifecycle"  — domain contradictions from `LifecycleStates.ttl`
                   (e.g. "out of use" vs "operational", "rain" vs "dry").
  * "paraphrase" — domain equivalences from `Paraphrase.ttl`
                   (e.g. "operate" ≡ "run", numeric probability buckets).
A fourth token, "conceptnet", is an ontology variant of "honk": the same
name_eq grounding run against the ConceptNet-only build
(`LaSSI/HOnK-cn-only.ttl`, RocksDB cache `cache_cn/`), for the degraded
single-source grounding arm. One ontology per process: run it in its own
invocation.

This is *grounding-only*: the LLM still produces the final implication score —
the ontologies never override it — so an `LLM#` vs `LLMHOnK#` comparison (and a
per-source ablation) is a clean measure of how much each symbolic source adds.

Wired as the `LLMHOnK#<model>[#<sources>]` FullText backend in `LaSSI/LaSSI.py`,
where `<sources>` is a '+'-joined subset of honk/lifecycle/paraphrase (omitted ⇒
all three).
"""

import re

from LaSSI.similarities.LLM import LLMPrompt

# Hardcoded fallback stoplist, used only when HOnK is unavailable so term
# extraction still drops the most common function words. When HOnK is ready the
# far richer ontology getters (pronouns/prepositions/conjunctions/...) are used.
_FALLBACK_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "to", "of", "in", "on", "at", "by", "for", "with", "and", "or", "but",
    "this", "that", "these", "those", "it", "its", "as", "from", "has", "have",
    "had", "will", "would", "no", "not", "do", "does", "did", "there", "their",
})

# Cap on the number of facts injected into the prompt, to bound prompt size and
# per-cell latency (the term cross-product is quadratic).
_MAX_FACTS = 15

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")

_ALL_SOURCES = frozenset({"honk", "lifecycle", "paraphrase"})

# "conceptnet" is an ontology *variant* of the honk source: the same grounding
# logic (name_eq over the term cross-product) run against a ConceptNet-only
# build of the ontology, for the degraded single-source grounding arm.
_VALID_TOKENS = _ALL_SOURCES | {"conceptnet"}

_HONK_TTL, _HONK_CACHE = "LaSSI/HOnK.ttl", "cache"
_CN_TTL, _CN_CACHE = "LaSSI/HOnK-cn-only.ttl", "cache_cn"

# The HOnKSingleton binds one ontology per process (fixed RocksDB store per
# cache dir), so record which TTL bootstrapped it and fail loudly on a mix.
_BOOTSTRAPPED_TTL = None


def parse_sources(spec) -> frozenset:
    """Parse a '+'-joined source spec (e.g. 'honk+lifecycle') into a validated
    frozenset. Empty/None ⇒ all three general sources (the headline config).
    Unknown tokens raise, so a typo cannot silently become a different
    condition. 'conceptnet' selects the ConceptNet-only ontology variant and
    cannot be combined with 'honk'."""
    if not spec:
        return _ALL_SOURCES
    if isinstance(spec, (set, frozenset)):
        chosen = {s.strip().lower() for s in spec}
    else:
        chosen = {s.strip().lower() for s in str(spec).replace(",", "+").split("+")}
    chosen.discard("")
    unknown = chosen - _VALID_TOKENS
    if unknown:
        raise ValueError(f"unknown grounding source(s) {sorted(unknown)}; "
                         f"valid: {sorted(_VALID_TOKENS)}")
    if "conceptnet" in chosen and "honk" in chosen:
        raise ValueError("'conceptnet' and 'honk' select different ontologies "
                         "and cannot be combined in one condition")
    return frozenset(chosen) if chosen else _ALL_SOURCES


def _ensure_honk(db_conf, ttl_path=_HONK_TTL, cache_path=_HONK_CACHE) -> bool:
    """Bring up the HOnK singleton if it isn't already (the FullText pipeline
    path returns before HOnK is initialised — see LaSSI/LaSSI.py:212-213).

    Mirrors LaSSI/LaSSI.py:215-221. Returns True if HOnK is ready afterwards,
    False otherwise (the caller then drops the HOnK source for that run).
    The singleton holds one ontology per process: mixing conditions that need
    different TTLs (honk vs conceptnet) in one process raises rather than
    silently grounding against the wrong graph.
    """
    global _BOOTSTRAPPED_TTL
    if _BOOTSTRAPPED_TTL is not None and _BOOTSTRAPPED_TTL != ttl_path:
        raise RuntimeError(
            f"HOnK singleton already bootstrapped from {_BOOTSTRAPPED_TTL}; "
            f"cannot re-ground against {ttl_path} in the same process. Run "
            f"this condition in a separate invocation.")
    try:
        from LaSSI.HOnK.HOnK import HOnKSingleton
        HOnKSingleton.instance()
        if HOnKSingleton.get() is None:
            HOnKSingleton.init(
                cache_path, db_conf.uname, db_conf.pw, db_conf.host, db_conf.port,
                False, ttl_path,
                rules_path="raw_data/logical_analysis.json",
            )
        if HOnKSingleton.isReady():
            _BOOTSTRAPPED_TTL = ttl_path
            return True
        return False
    except RuntimeError:
        raise
    except Exception as e:  # pragma: no cover - defensive bootstrap
        print(f"[LLMandHOnK] HOnK unavailable, dropping HOnK grounding source: "
              f"{type(e).__name__}: {e}")
        return False


class LLMHOnKPrompt(LLMPrompt):
    """LLM entailment scorer grounded with LaSSI ontology relations."""

    _GROUNDED_PROMPT_TEMPLATE = (
        "You are a strict logical reasoning assistant.\n\n"
        "Premise: '{premise}'\n"
        "Consequence: '{consequence}'\n\n"
        "Knowledge base facts (from the LaSSI ontologies):\n"
        "{evidence}\n\n"
        "Use the knowledge base facts above as verified background knowledge. "
        "Treat them as ground truth when they are relevant, but rely on your own "
        "judgment for anything they do not cover.\n"
        "Analyze if the premise implies the consequence. "
        "Respond ONLY in valid JSON using the following format:\n"
        "{{\n"
        "  \"reasoning\": \"Briefly explain your logic here\",\n"
        "  \"implication_score\": 1.0 (if fully true), 0.5 (if partially true), or 0.0 (if false)\n"
        "}}"
    )

    _NO_FACTS = "No ontology relations found between the salient terms."

    def __init__(self, model_name="qwen2.5", db_conf=None, sources=None):
        super().__init__(model_name)
        self.db_conf = db_conf
        self.sources = parse_sources(sources)
        # HOnK is only needed (and only bootstrapped) for the honk-like
        # sources; 'conceptnet' is the same grounding against the
        # ConceptNet-only ontology build (its own TTL and RocksDB cache).
        self._honk_like = ("honk" in self.sources) or ("conceptnet" in self.sources)
        if "conceptnet" in self.sources:
            ttl_path, cache_path = _CN_TTL, _CN_CACHE
        else:
            ttl_path, cache_path = _HONK_TTL, _HONK_CACHE
        self._honk_ready = (
            _ensure_honk(db_conf, ttl_path, cache_path)
            if (self._honk_like and db_conf is not None) else False
        )
        self._stopwords = self._load_stopwords()
        self._para_vocab = None       # lazily built phrase vocab for Paraphrase.ttl
        self._lifecycle_vocab = None  # lazily built phrase vocab for LifecycleStates.ttl
        self._words_cache = {}        # sentence -> set of tokens + verb-lemmas
        self._lemmatizer = None       # lazily constructed WordNet lemmatizer

    # -- term / phrase extraction ---------------------------------------

    def _load_stopwords(self) -> frozenset:
        """Function-word set used to drop non-content tokens. Built from HOnK's
        lexical getters when available, else the small hardcoded fallback."""
        if not self._honk_ready:
            return _FALLBACK_STOPWORDS
        try:
            from LaSSI.HOnK.HOnK import HOnKSingleton
            kb = HOnKSingleton.get()
            words = set(_FALLBACK_STOPWORDS)
            for getter in ("getPronouns", "getPersonalPronouns",
                           "getRelativePronouns", "getPrepositions",
                           "getConjunctions", "getSemiModalVerbs"):
                fn = getattr(kb, getter, None)
                if fn is not None:
                    words.update(w.lower() for w in (fn() or set()) if isinstance(w, str))
            return frozenset(words)
        except Exception:
            return _FALLBACK_STOPWORDS

    def _terms(self, sentence: str) -> list:
        """Salient terms of a sentence: content unigrams plus adjacent content
        bigrams (so multi-word ontology entries like 'bus station' are found).
        Order-preserving and de-duplicated. Used for the general-vocabulary
        HOnK source (not the fixed-phrase TTL sources, which are scanned)."""
        tokens = [t.lower() for t in _TOKEN_RE.findall(sentence or "")]
        content = [t for t in tokens if t not in self._stopwords and len(t) > 1]
        terms, seen = [], set()
        for t in content:
            if t not in seen:
                seen.add(t)
                terms.append(t)
        for a, b in zip(content, content[1:]):
            bigram = f"{a} {b}"
            if bigram not in seen:
                seen.add(bigram)
                terms.append(bigram)
        return terms

    def _sentence_words(self, sentence: str) -> set:
        """Set of single-word match keys for a sentence: surface tokens plus
        their verb lemmas, so a TTL lemma ('operate', 'run') matches an
        inflected surface form ('operating', 'running'). Falls back to bare
        tokens if the WordNet lemmatizer is unavailable. Cached per sentence."""
        s = (sentence or "").lower()
        if s in self._words_cache:
            return self._words_cache[s]
        tokens = set(_TOKEN_RE.findall(s))
        words = set(tokens)
        if self._lemmatizer is None:
            try:
                from nltk.stem import WordNetLemmatizer
                self._lemmatizer = WordNetLemmatizer()
            except Exception:
                self._lemmatizer = False  # mark as unavailable
        if self._lemmatizer:
            try:
                words.update(self._lemmatizer.lemmatize(t, "v") for t in tokens)
            except Exception:
                pass
        self._words_cache[s] = words
        return words

    def _scan_phrases(self, sentence: str, vocab) -> list:
        """Phrases from `vocab` (a set of lowercased labels) that occur in the
        sentence. Single-word labels match surface tokens or their verb lemmas;
        multi-word labels ('out of use', 'until further notice') match on word
        boundaries. Longer phrases first so the more specific label wins."""
        if not vocab:
            return []
        s = (sentence or "").lower()
        words = self._sentence_words(sentence)
        found = []
        for p in vocab:
            if not p:
                continue
            if " " in p or "-" in p:
                if re.search(r"(?<!\w)" + re.escape(p) + r"(?!\w)", s):
                    found.append(p)
            elif p in words:
                found.append(p)
        found.sort(key=len, reverse=True)
        return found

    # -- per-source fact builders ---------------------------------------

    def _honk_facts(self, premise: str, consequence: str) -> list:
        """HOnK name_eq verdicts over the term cross-product, as English facts."""
        if not self._honk_ready:
            return []
        try:
            from LaSSI.HOnK.HOnK import HOnKSingleton, CasusHappening
            kb = HOnKSingleton.get()
        except Exception:
            return []
        phrasing = {
            CasusHappening.EQUIVALENT: 'is equivalent to',
            CasusHappening.EXCLUSIVES: 'contradicts',
            CasusHappening.GENERAL_IMPLICATION: 'is a kind of',
            CasusHappening.INSTANTIATION_IMPLICATION: 'is a kind of',
            CasusHappening.LOSE_SPEC_IMPLICATION: 'implies',
            CasusHappening.MISSING_1ST_IMPLICATION: 'is implied by',
        }
        facts, seen = [], set()
        for pt in self._terms(premise):
            for ct in self._terms(consequence):
                if pt == ct:
                    continue
                try:
                    verdict = kb.name_eq(pt, ct)
                except Exception:
                    continue
                rel = phrasing.get(verdict)
                if rel is None:
                    continue
                key = (pt, ct, rel)
                if key not in seen:
                    seen.add(key)
                    facts.append(f'- "{pt}" {rel} "{ct}"')
        return facts

    def _lifecycle_facts(self, premise: str, consequence: str) -> list:
        """Domain status contradictions / agreements from LifecycleStates.ttl:
        two phrases on the same dimension contradict iff their (non-descriptive)
        partitions differ, and agree iff they match."""
        try:
            from LaSSI.HOnK.TBox.LifecycleManager import (
                _get_lifecycle_phrases, _DESCRIPTIVE_PARTITION)
            phrases = _get_lifecycle_phrases()
        except Exception:
            return []
        if not phrases:
            return []
        if self._lifecycle_vocab is None:
            self._lifecycle_vocab = set(phrases.keys())

        p_phr = self._scan_phrases(premise, self._lifecycle_vocab)
        c_phr = self._scan_phrases(consequence, self._lifecycle_vocab)
        facts, seen = [], set()
        for pt in p_phr:
            for ct in c_phr:
                if pt == ct:
                    continue
                for (dim_p, part_p) in phrases.get(pt, ()):
                    for (dim_c, part_c) in phrases.get(ct, ()):
                        if dim_p != dim_c:
                            continue
                        if part_p == _DESCRIPTIVE_PARTITION or part_c == _DESCRIPTIVE_PARTITION:
                            continue
                        # Dedup on the semantic key so surface variants of the
                        # same partition (e.g. "concluded"/"conclude") collapse.
                        key = (pt, dim_p, part_p, part_c)
                        if key in seen:
                            continue
                        seen.add(key)
                        dim_pretty = dim_p.replace("_", " ")
                        if part_p != part_c:
                            facts.append(f'- "{pt}" is incompatible with "{ct}" '
                                         f'(both describe {dim_pretty}, but assert opposite states)')
                        else:
                            facts.append(f'- "{pt}" and "{ct}" both indicate the same '
                                         f'{dim_pretty} state')
        return facts

    def _paraphrase_facts(self, premise: str, consequence: str) -> list:
        """Domain equivalences from Paraphrase.ttl: two phrases that belong to
        the same ParaphraseConcept denote the same idea."""
        try:
            from LaSSI.HOnK.TBox.ParaphraseManager import (
                _get_paraphrase_concepts, _paraphrase_match)
            concepts = _get_paraphrase_concepts()
        except Exception:
            return []
        if not concepts:
            return []
        if self._para_vocab is None:
            vocab = set()
            for bucket in concepts.values():
                for (_form, phrase) in bucket.get("members", ()):
                    if phrase:
                        vocab.add(phrase.strip().lower())
            self._para_vocab = vocab

        p_phr = self._scan_phrases(premise, self._para_vocab)
        c_phr = self._scan_phrases(consequence, self._para_vocab)
        facts, seen = [], set()
        for pt in p_phr:
            for ct in c_phr:
                if pt == ct:
                    continue
                try:
                    if not _paraphrase_match(pt, ct):
                        continue
                except Exception:
                    continue
                key = frozenset((pt, ct))  # symmetric — render once
                if key not in seen:
                    seen.add(key)
                    facts.append(f'- "{pt}" means the same as "{ct}"')
        return facts

    # -- evidence assembly ----------------------------------------------

    def _evidence(self, premise: str, consequence: str) -> str:
        """Combined evidence block, highest-signal sources first (lifecycle
        contradictions, then paraphrase equivalences, then HOnK relations),
        capped at `_MAX_FACTS`."""
        facts, seen = [], set()
        builders = []
        if "lifecycle" in self.sources:
            builders.append(self._lifecycle_facts)
        if "paraphrase" in self.sources:
            builders.append(self._paraphrase_facts)
        if self._honk_like:
            builders.append(self._honk_facts)
        for build in builders:
            for fact in build(premise, consequence):
                if fact not in seen:
                    seen.add(fact)
                    facts.append(fact)
                    if len(facts) >= _MAX_FACTS:
                        return "\n".join(facts)
        return "\n".join(facts) if facts else self._NO_FACTS

    # Backward-compatible alias (earlier smoke tests / callers used this name).
    def _honk_evidence(self, premise: str, consequence: str) -> str:
        return self._evidence(premise, consequence)

    # -- prompt override (reuses LLMPrompt's _dispatch plumbing) ---------

    def _build_prompt(self, premise: str, consequence: str) -> str:
        evidence = self._evidence(premise, consequence)
        return self._GROUNDED_PROMPT_TEMPLATE.format(
            premise=premise, consequence=consequence, evidence=evidence)
