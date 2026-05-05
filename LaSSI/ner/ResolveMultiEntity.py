# # Prallelized version:
# __author__ = "Oliver R. Fox, Giacomo Bergami"
# __copyright__ = "Copyright 2024, Oliver R. Fox, Giacomo Bergami"
# __credits__ = ["Oliver R. Fox, Giacomo Bergami"]
# __license__ = "GPL"
# __version__ = "2.0"
# __maintainer__ = "Oliver R. Fox, Giacomo Bergami"
# __status__ = "Production"
#
# import itertools
# import multiprocessing
# from concurrent.futures import ThreadPoolExecutor
# from typing import List
#
# from LaSSI.similarities.levenshtein import lev
# from LaSSI.structures.kernels.Sentence import lemmatize_verb
# from LaSSI.structures.meuDB.meuDB import MeuDBEntry
#
#
# # from gsmtosimilarity.TwoGrams import TwoGramSetSimilarity
# # from gsmtosimilarity.levenshtein import lev
#
#
# def build_loc_result(text, type, start_char, end_char, monad, conf, id, qa):
#     if isinstance(id, str):
#         yield MeuDBEntry(text, type, start_char, end_char, monad, conf, id, qa)
#     from collections.abc import Iterable
#     if isinstance(id, Iterable):
#         for x in id:
#             yield MeuDBEntry(text, type, start_char, end_char, monad, conf, x, qa)
#     else:
#         yield MeuDBEntry(text, type, start_char, end_char, monad, conf, str(id), qa)
#
# import asyncio
#
# def background(f):
#     def wrapped(*args, **kwargs):
#         return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)
#
#     return wrapped
#
# class ResolveMultiNamedEntity:
#
#     def __init__(self, threshold, forinsert, qa, parmo=None, trustworthiness_source=1.0):
#         self.trustworthiness_source = trustworthiness_source
#         self.parmo = parmo
#         self.threshold = threshold
#         self.forinsert = forinsert
#         self.result = []
#         self.s = None
#         self.fa = None
#         self.qa = qa
#
#     def _test(self, current, rest, k, v, start, end, type: str | List[str]):
#         if len(rest) == 0:
#             if k >= self.forinsert:
#                 if isinstance(type, list):
#                     type = self.parmo.most_specific_type(type)
#                 yield from build_loc_result(current, type, start, end, v, k * self.trustworthiness_source, v, self.qa)
#         else:
#             next = current + " " + rest[0][0]
#             val = lev(next.lower(), v.lower())
#             if val < k:
#                 if k >= self.forinsert:
#                     if isinstance(type, list):
#                         type = self.parmo.most_specific_type(type)
#                     yield from build_loc_result(current, type, start, end, v, k * self.trustworthiness_source, v,
#                                               self.qa)
#             else:
#                 yield from self._test(next, rest[1:], val, v, start, rest[0][2], type)
#
#
#     def start(self, stringa, s, fa, nlp, type):
#         self.s = s
#         self.fa = fa
#         self.result.clear()
#         self.holding_futures = []
#         cpus = int(multiprocessing.cpu_count() / 3 * 2)
#         with ThreadPoolExecutor(max_workers=cpus) as executor:
#             future = executor.submit(pow, 323, 1235)
#             # print(future.result())
#             for sentence in nlp(stringa).sentences:
#                 # List of lemmatized and non-lemmatized words from sentence
#                 ls = [(token.text, token.start_char, token.end_char) for token in sentence.tokens] + [
#                     (lemmatize_verb(token.text), token.start_char, token.end_char) for token in sentence.tokens]
#                 if type is None:
#                     for i in range(len(ls)):
#                         self.holding_futures.append(executor.submit(self.untyped_match, i, ls, s))
#                 else:
#                     for i in range(len(ls)):
#                         self.holding_futures.append(executor.submit(self.typed_match, i, ls, s, type))
#
#         self.result = [x.result() for x in self.holding_futures]
#         return itertools.chain.from_iterable(self.result)
#
#     def typed_match(self, i, ls, s, type):
#         m = s.fuzzyMatch(self.threshold, ls[i][0])
#         for k, v in m.items():
#             for candidate in v:
#                 # cand = s.get(candidate)
#                 newK = lev(ls[i][0].lower(), candidate.lower())
#                 if newK >= self.threshold:
#                     yield from self._test(ls[i][0], ls[i + 1:], newK, candidate, ls[i][1], ls[i][2], type)
#                 else:
#                     yield from []
#         if len(m)==0:
#             yield from []
#
#     def untyped_match(self, i, ls, s):
#         m = s.typedFuzzyMatch(self.threshold, ls[i][0])
#         for k, v in m.items():
#             for candidate, candidate_type in v:
#                 # cand = s.get(candidate)
#                 newK = lev(ls[i][0].lower(), candidate.lower())
#                 if newK >= self.threshold:
#                     yield from self._test(ls[i][0], ls[i + 1:], newK, candidate, ls[i][1], ls[i][2],
#                                         [candidate_type])
#                 else:
#                     yield from []
#         if len(m)==0:
#             yield from []


__author__ = "Oliver R. Fox, Giacomo Bergami"
__copyright__ = "Copyright 2024, Oliver R. Fox, Giacomo Bergami"
__credits__ = ["Oliver R. Fox, Giacomo Bergami"]
__license__ = "GPL"
__version__ = "2.0"
__maintainer__ = "Oliver R. Fox, Giacomo Bergami"
__status__ = "Production"

import itertools
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from typing import Iterable

from LaSSI.external_services.Services import Services
from LaSSI.external_services.utilities.FuzzyStringMatchDatabase import DBFuzzyStringMatching, FuzzyStringMatchDatabase
from LaSSI.ner.string_functions import lemmatize_verb
from LaSSI.similarities.levenshtein import lev
from LaSSI.structures.meuDB.meuDB import MeuDBEntry

def logger_func(x):
    return


def _build_loc_result_worker(text, type, start_char, end_char, monad, conf, id_val, src):
    if isinstance(id_val, str):
        return [MeuDBEntry(text, type, start_char, end_char, monad, conf, id_val, src)]
    if isinstance(id_val, Iterable):
        return [MeuDBEntry(text, type, start_char, end_char, monad, conf, x, src) for x in id_val]
    return [MeuDBEntry(text, type, start_char, end_char, monad, conf, str(id_val), src)]


def test(current, rest, k, v, start, end, type_val, forinsert, trustworthiness_source, src, parmo):
    results = []

    # No more tokens to check in the sequence
    if not rest:
        if k >= forinsert:
            if isinstance(type_val, list):
                type_val = parmo.most_specific_type(type_val) if parmo else (type_val[0] if type_val else "None")
            results.extend(
                _build_loc_result_worker(current, type_val, start, end, v, k * trustworthiness_source, v, src)
            )
        return results

    # Recursive step
    next_chunk = current + " " + rest[0][0]
    val = lev(next_chunk.lower(), v.lower())

    if val < k:
        # Similarity dropped, so chain ends, current match might still be valid
        if k >= forinsert:
            if isinstance(type_val, list):
                type_val = parmo.most_specific_type(type_val) if parmo else (type_val[0] if type_val else "None")
            results.extend(
                _build_loc_result_worker(current, type_val, start, end, v, k * trustworthiness_source, v, src)
            )
    else:
        # Similarity is good, continue the recursive check with the next token
        results.extend(
            test(next_chunk, rest[1:], val, v, start, rest[0][2], type_val, forinsert,
                 trustworthiness_source, src, parmo)
        )
    return results

services = None

def process_sentence(args):
    tokens_data, s, threshold, forinsert, src, parmo, trustworthiness_source, type_info = args

    if s is None and parmo is None:
        global services
        if services is None:
            services = Services(logger=logger_func)

        s = DBFuzzyStringMatching(services.postgres, "honk")
        parmo = services.getHOnK()

    sentence_results = []

    ls = [(text, start, end) for text, start, end in tokens_data] + \
         [(lemmatize_verb(text), start, end) for text, start, end in tokens_data]

    half_len = len(ls) // 2
    for i in range(len(ls)):
        # Try matching n-grams (up to 3 words) starting at i
        # But ensure we only combine tokens from the same half (original vs lemmatized)
        max_n = 3
        current_limit = half_len if i < half_len else len(ls)
        
        for n in range(1, min(max_n + 1, current_limit - i + 1)):
            n_gram_tokens = ls[i:i+n]
            text = " ".join([t[0] for t in n_gram_tokens])
            start_char = n_gram_tokens[0][1]
            end_char = n_gram_tokens[-1][2]
            term = text.lower()

            if type_info is None:
                m = s.typedFuzzyMatch(threshold, term)
                # If term is also a noun, avoid mis-tagging capitalized common nouns as LOC/GPE
                has_noun_match = any(cand_type.lower() == 'noun' for cands in m.values() for _, cand_type in cands)

                for k, v_list in m.items():
                    for candidate, candidate_type in v_list:
                        # Reject GPE/LOC matches when the original text starts lowercase
                        # but the candidate is a proper noun (capitalised in DB) — e.g. "nice" → "Nice".
                        # Allow generic location terms whose DB entry is also lowercase (e.g. "city centre").
                        # AND when it's capitalized at the start of the sentence but we have a common noun match
                        # AND the candidate itself is not capitalized (suggesting it's not a proper noun)
                        if candidate_type in {"GPE", "LOC"}:
                            if not text[0].isupper() and candidate[0].isupper():
                                continue
                            if has_noun_match and start_char == 0 and not candidate[0].isupper():
                                continue
                        newK = lev(term, candidate.lower())
                        if newK >= threshold:
                            sentence_results.extend(
                                test(
                                    text, ls[i + n:], newK, candidate, start_char, end_char,
                                    [candidate_type], forinsert, trustworthiness_source, src, parmo
                                )
                            )
            else:
                m = s.fuzzyMatch(threshold, term)
                for k, v_list in m.items():
                    for candidate in v_list:
                        newK = lev(term, candidate.lower())
                        if newK >= threshold:
                            sentence_results.extend(
                                test(
                                    text, ls[i + n:], newK, candidate, start_char, end_char,
                                    type_info, forinsert, trustworthiness_source, src, parmo
                                )
                            )
    return sentence_results


class ResolveMultiNamedEntity:

    def __init__(self, threshold, forinsert, src, parmo=None, trustworthiness_source=1.0):
        self.trustworthiness_source = trustworthiness_source
        self.parmo = parmo
        self.threshold = threshold
        self.forinsert = forinsert
        self.src = src
        self.s = None
        self.fa = None

    def start(self, stringa, s, fa, nlp, type):
        self.s = s
        self.fa = fa

        sentences_data = [
            [(token.text, token.start_char, token.end_char) for token in sent.tokens]
            for sent in nlp(stringa).sentences
        ]

        all_results = []

        if multiprocessing.get_start_method() == "spawn" and False:  # Leaving for now as appears it might be worse than just doing it in series.
            task_args = zip(
                sentences_data,
                itertools.repeat(None),
                itertools.repeat(self.threshold),
                itertools.repeat(self.forinsert),
                itertools.repeat(self.src),
                itertools.repeat(None),
                itertools.repeat(self.trustworthiness_source),
                itertools.repeat(type)
            )

            with ProcessPoolExecutor(max_workers=8) as executor:
                results_from_processes = executor.map(process_sentence, task_args)
                all_results = list(itertools.chain.from_iterable(results_from_processes))
        else:
            for sentence_tokens in sentences_data:
                args = (
                    sentence_tokens,
                    self.s,
                    self.threshold,
                    self.forinsert,
                    self.src,
                    self.parmo,
                    self.trustworthiness_source,
                    type
                )

                results_for_sentence = process_sentence(args)
                all_results.extend(results_for_sentence)

        return all_results
