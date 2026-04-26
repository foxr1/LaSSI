__author__ = "Oliver Robert Fox, Giacomo Bergami"
__copyright__ = "Copyright 2020, Giacomo Bergami"
__credits__ = ["Oliver Robert Fox", "Giacomo Bergami"]
__license__ = "GPL"
__version__ = "2.0"
__maintainer__ = "Giacomo Bergami"
__email__ = "bergamigiacomo@gmail.com"
__status__ = "Production"

from concurrent.futures import ProcessPoolExecutor
from LaSSI.phases.ResolveSingleSentence import process_sentence_worker
from LaSSI.structures.meuDB.meuDB import MeuDB
from LaSSI.tests.benchmark import Benchmark


class ResolveBasicTypes:
    def __init__(self, recall_threshold: float, precision_threshold: float, disable_a_priori: bool, use_multiprocessing: bool, disable_fuzzy_honk: bool = False):
        from LaSSI.external_services.Services import Services
        self.disable_a_priori = disable_a_priori
        self.disable_fuzzy_honk = disable_fuzzy_honk
        self.recall_threshold = recall_threshold
        self.precision_threshold = precision_threshold
        self.services = Services.getInstance()
        self.stanza_service = self.services.getStanzaNLP()
        self.sentences_benchmark = Benchmark()
        self.use_multiprocessing = use_multiprocessing

    def resolve_basic_types(self, list_sentences):
        if self.disable_a_priori:
            return [MeuDB(sentence, []) for sentence in list_sentences]

        all_time_units = self.services.resolveTimeUnits(list_sentences)
        db = [None] * len(list_sentences)

        tasks_args = []
        for i, sentence in enumerate(list_sentences):
            args = (
                i,
                sentence,
                all_time_units[i],
                self.recall_threshold,
                self.precision_threshold,
                self.disable_fuzzy_honk
            )
            tasks_args.append(args)

        if self.use_multiprocessing:
            with ProcessPoolExecutor(max_workers=8) as executor:
                results_iterator = executor.map(process_sentence_worker, tasks_args)

                for result in results_iterator:
                    try:
                        idx, meu_db_obj, benchmark_data = result
                        db[idx] = meu_db_obj
                        # self.sentences_benchmark.add_row(idx, "Generating meuDB", end_time - start_time)
                        print(benchmark_data)
                    except Exception as exc:
                        print(f"A task generated an exception: {exc}")
        else:
            for args in tasks_args:
                try:
                    result = process_sentence_worker(args)
                    idx, meu_db_obj, benchmark_data = result
                    db[idx] = meu_db_obj
                    print(benchmark_data)
                except Exception as exc:
                    print(f"A task generated an exception: {exc}")
        return db


def ExplainTextWithNER(self, sentences, disable_fuzzy_honk=False):
    return ResolveBasicTypes(self.recall_threshold, self.precision_threshold, self.disable_a_priori, self.use_multiprocessing, disable_fuzzy_honk).resolve_basic_types(sentences)
