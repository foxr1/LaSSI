import logging
import re
import time as ti

from LaSSI.similarities.levenshtein import lev
from LaSSI.structures.meuDB.meuDB import MeuDBEntry, MeuDB

services = None
stanza_service = None

# ISO 8601 datetime: 2026-04-14T14:00Z, 2026-04-14T14:00:00, 2026-04-14T14:00+02:00, etc.
# CoreNLP tokenises this into multiple chunks at `T` and the dashes, so add an
# explicit DATE MEU spanning the whole thing — TypeResolver.mergeMeuNodes folds
# the chunks back into a single node when a MEU contains them.
_ISO_8601_DATETIME = re.compile(
    r"\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?\b"
)
# Date-only ISO 8601: 2026-04-27 (no time component).
_ISO_8601_DATE_ONLY = re.compile(
    r"\b\d{4}-\d{2}-\d{2}(?!T\d)\b"
)



def logger_func(x):
    return

def process_sentence_worker(args):
    from LaSSI.external_services.Services import Services
    
    # Check if we have 5 or 6 arguments
    if len(args) == 6:
        idx, sentence, with_time, recall_threshold, precision_threshold, disable_fuzzy_honk = args
    else:
        idx, sentence, with_time, recall_threshold, precision_threshold = args
        disable_fuzzy_honk = False

    global services, stanza_service
    if services is None:
        services = Services.getInstance(logger=logger_func)
        stanza_service = services.getStanzaNLP()

    print(f"Processing sentence: {sentence}")
    start_time = ti.time()

    multi_entity_unit = []

    _iso_spans = set()
    for m in _ISO_8601_DATETIME.finditer(sentence):
        text = m.group(0)
        _iso_spans.add((m.start(), m.end()))
        multi_entity_unit.append(MeuDBEntry(
            text, "DATE", m.start(), m.end(),
            text, 1.0, text, "ISO8601"))
    for m in _ISO_8601_DATE_ONLY.finditer(sentence):
        text = m.group(0)
        span = (m.start(), m.end())
        # Skip if a longer datetime span starts at the same position
        if any(s <= m.start() and e >= m.end() for (s, e) in _iso_spans):
            continue
        multi_entity_unit.append(MeuDBEntry(
            text, "DATE", m.start(), m.end(),
            text, 1.0, text, "ISO8601"))

    if not disable_fuzzy_honk:
        multi_entity_unit.extend(services.getFuzzyHOnK().resolve_u(
            recall_threshold, precision_threshold, sentence))

    # multi_entity_unit.extend(services.getGeoNames().resolve_u(
    #     recall_threshold, precision_threshold, sentence, "GPE"))
    #
    # multi_entity_unit.extend(services.getConcepts().resolve_u(
    #     recall_threshold, precision_threshold, sentence, "ENTITY"))

    for time_info in with_time:
        time_entry = MeuDBEntry.from_dict_with_src(time_info, "SUTime")
        multi_entity_unit.append(time_entry)

    results = stanza_service(sentence)
    org_entities_for_replacement = []

    for ent in results.ents:
        monad = ent.text #.replace(" ", "") # TODO: This is affecting sentences from NEET, what was this used for before?
        if ent.type == "ORG":
            org_entities_for_replacement.append((ent.text, monad))

        # TODO: Might not need this hardcode but leaving for now...
        ent_type = "LOC" if ent.type == "FAC" else ent.type

        multi_entity_unit.append(MeuDBEntry(
            ent.text, ent_type, ent.start_char, ent.end_char,
            monad, lev(monad.lower(), ent.text.lower()), monad, "Stanza"))

    for sent in results.sentences:
        for word in sent.words:
            if word.pos.lower() == 'verb':
                multi_entity_unit.append(MeuDBEntry(
                    word.text, word.pos.lower(), word.start_char,
                    word.end_char, word.lemma, 1.0, word.lemma, "Stanza"))

    modified_sentence = sentence
    for original, replacement in org_entities_for_replacement:
        modified_sentence = modified_sentence.replace(original, replacement)

    end_time = ti.time()
    benchmark_data = (idx, "Generating meuDB", end_time - start_time)

    print(f"Finished processing: {sentence}")
    return idx, MeuDB(modified_sentence, multi_entity_unit), benchmark_data