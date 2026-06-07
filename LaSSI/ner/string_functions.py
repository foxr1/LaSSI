import re
from functools import lru_cache

from LaSSI.external_services.Services import Services

HONK_NS = "https://ofox.co.uk/honk#"

negations = {'not', 'no'}

# Dependency markers (case / preposition tokens) are stored on a node under
# float-formatted *position* keys, e.g. "10.000000". Several passes (graph
# construction, the structural rewrites, the kernel rewriter) need to recognise
# these; this is the single shared predicate so the pattern is not re-implemented
# per module.
POSITION_KEY_RE = re.compile(r"^\d+(?:\.\d+)?$")


def is_position_key(key) -> bool:
    """True when *key* is a position-keyed property name (e.g. "10.000000")."""
    try:
        return bool(POSITION_KEY_RE.match(str(key)))
    except (TypeError, ValueError):
        return False

@lru_cache(maxsize = 1024)
def is_label_verb(edge_label_name):
    edge_label_name = lemmatize_verb(edge_label_name).lower()
    honk_types = {str(x)[len(HONK_NS):] for x in
                  Services.getInstance().getHOnK().typeOf(edge_label_name)}
    is_verb = any(map(lambda x: 'Verb' in x, honk_types))
    return is_verb

def does_string_have_negations(edge_label_name):
    return bool(re.search(r"\b(" + "|".join(re.escape(neg) for neg in negations) + r")\b", edge_label_name, flags=re.IGNORECASE))

_APOSTROPHE_VARIANTS = ("'", "’", "ʼ")  # straight, curly, modifier letter


def normalise_apostrophes(name):
    """Strip possessive/trailing apostrophes from `name` so ontology lookups
    match e.g. ``Nicholas' Church`` against ``Nicholas Church``.
    Returns the original string if no change applies."""
    if not isinstance(name, str) or not name:
        return name
    canonical = name
    for variant in _APOSTROPHE_VARIANTS:
        if variant != "'" and variant in canonical:
            canonical = canonical.replace(variant, "'")
    canonical = re.sub(r"'s\b", "s", canonical)
    canonical = re.sub(r"'(?=\s|$)", "", canonical)
    canonical = canonical.replace("'", "")
    return canonical


_ontology_class_suffix_terms_cache: frozenset | None = None

def _ontology_class_suffix_terms():
    """All facility / access-point / route / location nouns that HOnK knows
    about, as a frozenset of lowercase strings. These are the candidate
    type-suffixes that can be stripped from a compound surface form
    (e.g. `station`, `bridge`, `car park`, `metro station` if classed as
    such, etc.). Sourced from the noun lists loaded into HOnK from
    ``raw_data/nouns/`` — no hardcoded suffix list lives here."""
    global _ontology_class_suffix_terms_cache
    if _ontology_class_suffix_terms_cache:
        return _ontology_class_suffix_terms_cache
    try:
        honk = Services.getInstance().getHOnK()
    except Exception:
        return frozenset()
    out = set()
    for getter in (honk.getFacilityNouns, honk.getAccessPointNouns,
                   honk.getRouteNouns, honk.getLocationNouns):
        try:
            out.update(getter() or set())
        except Exception:
            continue
    result = frozenset(s.lower() for s in out if isinstance(s, str) and s)
    if result:
        _ontology_class_suffix_terms_cache = result
    return result


def class_suffix_variants(name):
    """Yield `name` plus prefix-only variants formed by iteratively stripping
    any HOnK-known noun class that appears as a whitespace-separated suffix.

    Concretely, given ``Haymarket Metro station``, if HOnK classes ``station``
    as a facility noun, the function yields both ``Haymarket Metro station``
    and ``Haymarket Metro``. If the ontology *also* classes ``metro`` as a
    facility/route noun, a further pass produces ``Haymarket`` — i.e. the
    behaviour of the old hardcoded ``metro_station_variants`` is recovered
    by populating the noun lists, with no Python changes needed.

    Comparison is case-insensitive against the lowercase ontology terms,
    but the returned variants preserve the input casing of the prefix."""
    if not isinstance(name, str) or not name:
        return {name} if name else set()
    suffixes = _ontology_class_suffix_terms()
    if not suffixes:
        return {name}
    variants = {name}
    queue = [name]
    while queue:
        candidate = queue.pop()
        c_lower = candidate.lower()
        for sfx in suffixes:
            sep_sfx = ' ' + sfx
            if c_lower.endswith(sep_sfx) and len(candidate) > len(sep_sfx):
                stripped = candidate[: -len(sep_sfx)].strip()
                if stripped and stripped not in variants:
                    variants.add(stripped)
                    queue.append(stripped)
    return variants


def surface_form_variants(name):
    """Combined surface-form candidates used by HOnK lookups.
    Includes the original, apostrophe-normalised form, and class-suffix
    stripped variants of both."""
    if not isinstance(name, str) or not name:
        return {name} if name else set()
    seeds = {name}
    apos = normalise_apostrophes(name)
    if apos and apos != name:
        seeds.add(apos)
    variants = set()
    for seed in seeds:
        variants |= class_suffix_variants(seed)
    return variants

def match_whole_word(w):
    return re.compile(r'\b({0})\b'.format(w), flags=re.IGNORECASE).search

@lru_cache(maxsize = 1024)
def lemmatize_verb(edge_label_name):
    if not edge_label_name or '?' in edge_label_name:
        return edge_label_name

    stNLP = Services.getInstance().getStanzaSTNLP()
    lemmatizer = Services.getInstance().getWTLemmatizer()

    try:
        lem_dict = stNLP(lemmatizer.lemmatize(edge_label_name, 'v')).to_dict()[0]
        lemmas = [token["lemma"] for token in lem_dict if token["upos"] != "AUX"]
        lemmatized_verb = " ".join(lemmas)
        return lemmatized_verb
    except (KeyError, IndexError):
        return edge_label_name

@lru_cache(maxsize = 1024)
def has_auxiliary(label):
    stNLP = Services.getInstance().getStanzaSTNLP()
    return any(map(lambda y: y["lemma"], filter(lambda x: x["upos"] == "AUX", stNLP(label).to_dict()[0])))


@lru_cache(maxsize = 1024)
def check_semi_modal(label):
    return len(Services.getInstance().getHOnK().getSemiModalVerbs().intersection({label})) != 0


@lru_cache(maxsize = 1024)
def lemmatize_sentence(text):
    stNLP = Services.getInstance().getStanzaSTNLP()
    doc = stNLP(text)
    lemmas = set()
    for sentence in doc.sentences:
        for word in sentence.words:
            if word.pos.lower() != 'aux':
                lemmas.add(word.lemma)
    return lemmas