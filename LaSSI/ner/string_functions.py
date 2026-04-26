import re
from functools import lru_cache

from LaSSI.external_services.Services import Services

HONK_NS = "https://ofox.co.uk/honk#"

negations = {'not', 'no'}

@lru_cache(maxsize = 1024)
def is_label_verb(edge_label_name):
    edge_label_name = lemmatize_verb(edge_label_name).lower()
    honk_types = {str(x)[len(HONK_NS):] for x in
                  Services.getInstance().getHOnK().typeOf(edge_label_name)}
    is_verb = any(map(lambda x: 'Verb' in x, honk_types))
    return is_verb

def does_string_have_negations(edge_label_name):
    return bool(re.search(r"\b(" + "|".join(re.escape(neg) for neg in negations) + r")\b", edge_label_name))

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