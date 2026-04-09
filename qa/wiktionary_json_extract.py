
short_to_long = {
    "adj": "Adjective",
    "adv": "Adverb",
    "verb": "Verb",
    "noun": "Noun",
    "prep": "Preposition",
    "pron": "Pronoun",
    "particle": "Particle",
    "num": "Numeral",
    "conj": "Conjunction",
    "det": "Determiner",
    "intj": "Interjection",
    "article": "Article",
    "abbrev":"Abbreviation"
}
long_to_short = {
    "Adjective": "adj",
    "adjective": "adj",
    "Adverb": "adv",
    "adverb": "adv",
    "Verb": "verb",
    "verb": "verb",
    "Noun": "noun",
    "noun": "noun",
    "Preposition": "prep",
    "preposition": "prep",
    "Pronoun":"pron",
    "pronoun":"pron",
    "Particle": "particle",
    "particle": "particle",
    "Numeral": "num",
    "numeral": "num",
    "Conjunction":"conj",
    "conjunction":"conj",
    "Proper noun": "noun",
    "Determiner":"det",
    "Interjection":"intj",
    "Article": "article",
    "Abbreviation": "abbrev",
    "abbreviation": "abbrev"
}

def extract_word(obj, language_code=None)->str:
    if (language_code is not None) and (("lang_code" not in obj) or obj["lang_code"] != language_code):
        return None
    return obj["word"]

def extract_information(obj, language_code=None):
    defaulted = []
    if (language_code is not None) and (("lang_code" not in obj) or obj["lang_code"] != language_code):
        return None
    form_of = obj["word"]
    pos = obj["pos"]
    word = form_of+"#"+pos
    yield tuple([form_of, "with_pos", word])
    prev_sense = {"meronym", "metonym", "holonym", "instance", "antonym", "synonym", "hypernym", "hyponym",
                  "meronyms", "metonyms", "holonyms", "instances", "antonyms", "synonyms", "hypernyms", "hyponyms",
                  "etymology", "hebrew", "romainan","greek", "german", "latin", "french", "catalan", "irish", "old","japanese","italian","dutch"}
    for inflection in obj.get("forms", defaulted):
        yield tuple([word, "inflection", inflection["form"]])
    yield from extract_fields(defaulted, word, obj)
    for sense_idx in range(len(obj["senses"])):
        sense = obj["senses"][sense_idx]
        full_ref = word+"#"+str(sense_idx)
        yield tuple([word, "with_sense", full_ref]) # I was trying to figure out what with_sense and with_pos can be mapped to. with_pos can be maybe be isa (cuz it isa "verb", e.g.)
        yield from extract_fields(defaulted, full_ref, sense)
        for link in sense.get("links", defaulted):
            if (link[1].startswith("w:") or "usage notes" in link[1].lower() or "usage_notes" in link[1].lower()):
                continue
            if ("#" not in link[1]):
                yield tuple([full_ref, "related_to", link[0]])
                yield tuple([link[0], "related_to", full_ref])
            else:
                t = tuple(link[1].split("#"))
                word, arg = t[0], t[-1]
                arg = arg.split("_")[0]
                asplit = arg.split(" ")
                if len(asplit) == 2:
                    if str(asplit[1]).isnumeric():
                        arg = asplit[0]
                if arg.split(" ")[0].lower() in prev_sense:
                    continue
                word = word.strip()
                if len(word) == 0:
                    word = link[0]
                if ("english" in arg.lower()) or "translingual" in arg.lower():
                    arg = None
                if arg is not None:
                    # if arg not in long_to_short:
                    #     # print("ERROR!")
                    arg = long_to_short.get(arg, None)
                    if arg is None:
                        mutchin = word
                    else:
                        mutchin = word+"#"+arg
                else:
                    mutchin = word
                yield tuple([full_ref, "related_to", mutchin])
                yield tuple([mutchin, "related_to", full_ref])


def extract_fields(defaulted, full_ref, sense):
    for form_of in sense.get("form_of", defaulted):
        yield tuple([full_ref, "form_of", form_of["word"]])
    for instance_of in sense.get("instances", defaulted):
        yield tuple([full_ref, "form_of", instance_of["word"]])
        yield tuple([full_ref, "instance_of", instance_of["word"]])
    for meronym in sense.get("meronyms", defaulted):
        yield tuple([full_ref, "part_of", meronym["word"]])
    for holonym in sense.get("holonyms", defaulted):
        yield tuple([holonym["word"], "part_of", full_ref])
    for antonym in sense.get("antonyms", defaulted):
        yield tuple([full_ref, "not_eq", antonym["word"]])
        yield tuple([antonym["word"], "not_eq", full_ref])
    for antonym in sense.get("synonyms", defaulted):
        yield tuple([full_ref, "eq", antonym["word"]])
        yield tuple([antonym["word"], "eq", full_ref])
    for coordinate_term in sense.get("coordinate_terms", defaulted):
        yield tuple([full_ref, "shares_isa_with", coordinate_term["word"]])
    for hypernym in sense.get("hypernyms", defaulted):
        yield tuple([full_ref, "isa", hypernym["word"]])
    for related in sense.get("related", defaulted):
        yield tuple([full_ref, "related_to", related["word"]])
        yield tuple([related["word"], "related_to", full_ref])
    for hyponym in sense.get("hyponyms", defaulted):
        yield tuple([hyponym["word"], "isa", full_ref])