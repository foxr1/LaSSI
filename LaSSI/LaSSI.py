__author__ = "Giacomo Bergami"
__copyright__ = "Copyright 2020, Giacomo Bergami"
__credits__ = ["Giacomo Bergami"]
__license__ = "GPL"
__version__ = "2.0"
__maintainer__ = "Giacomo Bergami"
__email__ = "bergamigiacomo@gmail.com"
__status__ = "Production"

import collections
import io
import json
import multiprocessing
import os.path
import time

import pkg_resources

from LaSSI.Configuration import SentenceRepresentation
from LaSSI.external_services.Services import Services
from LaSSI.external_services.utilities.DatabaseConfiguration import DatabaseConfiguration, load_db_configuration
from LaSSI.external_services.utilities.FuzzyStringMatchDatabase import FuzzyStringMatchDatabase
from LaSSI.external_services.web_cralwer.ScraperConfiguration import ScraperConfiguration
from LaSSI.files.JSONDump import json_dumps, obj_unmarshall, obj_pickle
from LaSSI.phases.ApplyGraphGrammars import ApplyGraphGrammars
from LaSSI.phases.CalculateMatrix import CalculateMatrix
from LaSSI.phases.GetGSMString import GetGSMString
from LaSSI.phases.LogicalRewriting import LogicalRewriting
from LaSSI.phases.ResolveBasicTypes import ExplainTextWithNER
from LaSSI.phases.SemanticGraphRewriting import SemanticGraphRewriting
from LaSSI.similarities.Classifier import Classifier
from LaSSI.similarities.graph_similarity import SimilarityScore
from LaSSI.structures.extended_fol.Formulae import formula_from_dict
from LaSSI.structures.internal_graph.Graph import Graph
from LaSSI.structures.internal_graph.InternalData import InternalRepresentation
from LaSSI.structures.meuDB.meuDB import MeuDB
from LaSSI.tests.benchmark import Benchmark
from LaSSI.utils.configurations import LegacySemanticConfiguration


def write_variable_to_file(dir, text):
    try:
        with open(dir, 'a') as file:
            file.write(str(text))
    except Exception as e:
        print(f"An error occurred: {e}")

def logger_func(x):
    return print(x)

class LaSSI():
    def __init__(self, dataset_name: str,
                 fuzzyDBs: str | DatabaseConfiguration,
                 transformation: SentenceRepresentation = SentenceRepresentation.Logical,
                 transformer='sentence-transformers/all-MiniLM-L6-v2',  # all-MiniLM-L6-v2 / all-MiniLM-L12-v2 / all-mpnet-base-v2 / all-roberta-large-v1 / RAG#colbert-ir/colbertv2.0"
                 sentences: ScraperConfiguration | str | collections.abc.Iterable = None,
                 logger=None,
                 web_dir=None,
                 recall_threshold=0.1,
                 precision_threshold=0.8,
                 force=False,
                 should_benchmark=True,
                 legacy_conf: LegacySemanticConfiguration = None,
                 disable_a_priori: bool = False,
                 run_ex_post: bool = True,
                 useId:bool = False,
                 use_multiprocessing=True,
                 ):
        self.use_multiprocessing = use_multiprocessing
        if use_multiprocessing:
            try:
                multiprocessing.set_start_method('spawn')
            except RuntimeError:
                pass
        self.useId = useId
        self.disable_a_priori = disable_a_priori
        if legacy_conf is None:
            self.legacy_conf = LegacySemanticConfiguration()
        else:
            self.legacy_conf = legacy_conf
        self.legacy_conf.HuggingFace = transformer
        self.string_rep_dir = None
        self.benchmarking_file = None
        self.run_ex_post = run_ex_post
        self.create_catabolites_dir(dataset_name)
        self.dataset_name = dataset_name
        tmp = f"{self.dataset_name}_clusters.txt"
        if os.path.isfile(tmp):
            self.clusters_file = tmp
        else:
            self.clusters_file = None
        import pathlib
        p = pathlib.Path(self.dataset_name)
        tmp = os.path.join(p.parent.absolute(), f"{p.stem}_matrix.json")
        # tmp = f"{self.dataset_name}_matrix.json"
        if os.path.isfile(tmp):
            self.matrix_file = tmp
        else:
            self.matrix_file = None
        self.web_dir = web_dir
        if logger is None:
            logger = logger_func
        self.logger = logger

        self.logger("init postgres")
        if not isinstance(fuzzyDBs, DatabaseConfiguration):
            fuzzyDBs = str(fuzzyDBs)
            fuzzyDBs = load_db_configuration(fuzzyDBs)
        # if hasattr(fuzzyDBs, "huggingface") and fuzzyDBs.huggingface is not None:
        #     self.legacy_conf.HuggingFace = fuzzyDBs.huggingface

        self.logger("init non-postgres services and the wrapper for the former...")
        self.initServices = Services.getInstance(self.logger)

        self.logger("init postgres services...")
        self.logger(" - Initialising the connection to the database")
        (FuzzyStringMatchDatabase
         .instance()
         .init(fuzzyDBs.db, fuzzyDBs.uname, fuzzyDBs.pw, fuzzyDBs.host, fuzzyDBs.port))

        self.logger(" - Loading the tab files or streaming those remotely, if required.")
        import tempfile
        for k, v in fuzzyDBs.fuzzy_dbs.items():
            self.logger(f" - Loading {k}.")
            FuzzyStringMatchDatabase.instance().create(k, v)

        try:
            from nltk.corpus import wordnet
            # Attempting to access a resource will trigger a LookupError if not downloaded
            wordnet.synsets('city')
            print("WordNet already downloaded.")
        except LookupError:
            print("WordNet not downloaded...")
            import nltk
            import ssl
            try:
                _create_unverified_https_context = ssl._create_unverified_context
            except AttributeError:
                pass
            else:
                ssl._create_default_https_context = _create_unverified_https_context
            nltk.download('wordnet')
            print("Downloaded WordNet.")

        if sentences is None:
            sentences = open(self.dataset_name, "r")
        self.sentences = sentences
        self.recall_threshold = recall_threshold
        self.precision_threshold = precision_threshold
        self.transformation = transformation
        self.full_transformation = transformation
        if self.disable_a_priori:
            if self.transformation == SentenceRepresentation.Logical:
                self.full_transformation = SentenceRepresentation.LogicalDisabledAPriori
            elif self.transformation == SentenceRepresentation.LogicalGraph:
                self.full_transformation = SentenceRepresentation.LogicalGraphDisabledAPriori
            elif self.transformation == SentenceRepresentation.SimpleGraph:
                self.full_transformation = SentenceRepresentation.SimpleGraphDisabledAPriori
        if self.transformation == SentenceRepresentation.LogicalDisabledAPriori:
            self.disable_a_priori = True
            self.transformation = SentenceRepresentation.Logical
        elif self.transformation == SentenceRepresentation.Logical:
            self.disable_a_priori = False
        elif self.transformation == SentenceRepresentation.SimpleGraphDisabledAPriori:
            self.disable_a_priori = True
            self.transformation = SentenceRepresentation.SimpleGraph
        elif self.transformation == SentenceRepresentation.SimpleGraph:
            self.disable_a_priori = False
        elif self.transformation == SentenceRepresentation.LogicalGraphDisabledAPriori:
            self.disable_a_priori = True
            self.transformation = SentenceRepresentation.LogicalGraph
        elif self.transformation == SentenceRepresentation.LogicalGraph:
            self.disable_a_priori = False
        self.force = force
        self.should_benchmark = should_benchmark
        self.logger("init file structure")
        from pathlib import Path
        self.catabolites = os.path.join("catabolites", self.catabolites_dir)
        self.catabolites_viz = os.path.join(self.catabolites, "viz")
        self.internals = os.path.join(self.catabolites, "internals.json")
        self.logical_rewriting = os.path.join(self.catabolites, "logical_rewriting.json")
        self.confusion_matrices = os.path.join(self.catabolites, "confusion_matrices_")
        Path(self.catabolites).mkdir(parents=True, exist_ok=True)
        Path(self.catabolites_viz).mkdir(parents=True, exist_ok=True)
        self.meuDB = os.path.join(self.catabolites, "meuDBs.json")
        self.gsmDB = os.path.join(self.catabolites, "gsmDB.txt")
        self.datagramdb_output = os.path.join(self.catabolites, "datagramdb_output.json")
        self.query_file = pkg_resources.resource_filename("LaSSI.resources", "gsm_query.txt")
        self.sc = None
        self.meu_dbs = None
        self.sentences_benchmark = Benchmark()


        from LaSSI.Parmenides.Parmenides import ParmenidesSingleton
        ParmenidesSingleton.instance()
        ## TODO: move parmenides.ttl to the resources
        ParmenidesSingleton.init("catabolites", fuzzyDBs.uname, fuzzyDBs.pw,
                                 fuzzyDBs.host, fuzzyDBs.port, False, "parmenides.ttl")
        self.initServices.setParmenides(ParmenidesSingleton.get())
        with tempfile.NamedTemporaryFile() as parmenides_tab:
            with open(parmenides_tab.name, 'w') as f:
                self.initServices.getParmenides().dumpTypedObjectsToTAB(f)
            FuzzyStringMatchDatabase.instance().create("parmenides", parmenides_tab.name, '(id integer NOT NULL, idx text, t text, type text)')  # Typed

    def create_catabolites_dir(self, dataset_name):
        from pathlib import Path
        self.catabolites_dir = Path(dataset_name).stem
        self.catabolites_of_dataset = os.path.join("catabolites", self.catabolites_dir)
        self.string_rep_dir = os.path.join(self.catabolites_of_dataset, "string_rep.txt")
        self.benchmarking_file = os.path.join("catabolites", "benchmark.csv")
        if os.path.exists(self.string_rep_dir):
            os.remove(self.string_rep_dir)
        if not os.path.exists(self.benchmarking_file):
            if not os.path.exists("catabolites"):
                os.makedirs("catabolites")
            write_variable_to_file(self.benchmarking_file, "Dataset,Loading sentences,Generating meuDB,"
                                                                "Loading meuDB,Generating gsmDB,Generating "
                                                                "rewritten graphs,Generating intermediate "
                                                                "representation,Generating logical representation")
            if self.run_ex_post:
                write_variable_to_file(self.benchmarking_file, ",Performing ex post explanation\n")
            else:
                write_variable_to_file(self.benchmarking_file, "\n")
        else:
            # If last line is not finished, add new line to ensure next benchmark is written to file correctly
            with open(self.benchmarking_file, 'r') as file:
                if file.readlines()[-1].rstrip('\n').endswith(','):
                    write_variable_to_file(self.benchmarking_file, "\n")

    def apply_graph_grammars(self, n):
        from PyDatagramDB import DatagramDB
        d = DatagramDB(self.gsmDB,
                       self.query_file,
                       self.catabolites_viz,
                       isSerializationFull=True,
                       opt_data_schema="pos\nSizeTAtt\nbegin\nSizeTAtt\nend\nSizeTAtt")
        d.run()
        L = []
        for result_graph_file in map(lambda x: os.path.join(self.catabolites_viz, str(x), "result.json"), range(n)):
            with open(result_graph_file, "r") as f:
                raw_json_graph = json.load(f)
                L.append(raw_json_graph)

        if self.web_dir is not None:
            import shutil
            dataset_folder = os.path.join(self.web_dir, "dataset", "data")  # f"{self.web_dir}/dataset/data"
            if os.path.exists(dataset_folder):
                shutil.rmtree(dataset_folder)
            shutil.copytree(self.catabolites_viz, dataset_folder)

        return L

    def _internal_graph(self, gsm_list):
        internal_representations = []
        for idx, (graph, meu_db) in enumerate(zip(gsm_list, self.meu_dbs)):
            start = time.time()
            from LaSSI.structures.provenance.GraphProvenance import GraphProvenance
            g = GraphProvenance(graph, meu_db, self.transformation == SentenceRepresentation.SimpleGraph)
            self.logger(f"{meu_db.first_sentence}")
            write_variable_to_file(self.string_rep_dir, meu_db.first_sentence)
            internal_graph = g.internal_graph()
            final_form = internal_graph
            if self.transformation == SentenceRepresentation.Logical:
                final_form = g.sentence()
                write_variable_to_file(self.string_rep_dir, f" ⇒ {final_form.to_string()}\n")
            internal_representations.append(final_form)
            end = time.time()
            self.sentences_benchmark.add_row(idx, "Sentence length", len(graph))
            self.sentences_benchmark.add_row(idx, "Generating intermediate representation", end - start)
        return internal_representations

    def _logical_rewriting(self, intermediate_representations):
        # logical_representations = []
        #
        # Loop over each Sentence
        # for intermediate_representation in intermediate_representations:
        #     for sentence in intermediate_representation.sentences:
        #     logical_representations.append(rewrite_kernels(intermediate_representation))
        from LaSSI.structures.extended_fol.rewrite_kernels import rewrite_kernels

        rewritten_kernels = []
        for idx, x in enumerate(intermediate_representations):
            start = time.time()
            rewritten_kernels.append(rewrite_kernels(x, self.meu_dbs[idx], self.useId))
            end = time.time()
            self.sentences_benchmark.add_row(idx, "Generating logical representation", end - start)
        return rewritten_kernels
        # return [rewrite_kernels(x, self.meu_dbs[idx]) for idx, x in enumerate(intermediate_representations)]

    def graph_with_logic_similarity(self, x: Graph, y: Graph) -> float:
        if self.sc is None:
            self.sc = SimilarityScore(self.legacy_conf)
        dist = self.sc.graph_distance(x, y) * 1.0
        return 1.0 - dist  # / (1 + dist)

    def fulltext_similarity(self, x: str, y: str) -> float:
        if self.sc is None:
            self.sc = SimilarityScore(self.legacy_conf)
        return self.sc.string_similarity(x, y)



    def _calculate_matrix(self, obj_list):
        matrices = None
        if self.transformation == SentenceRepresentation.FullText and self.legacy_conf.HuggingFace.startswith("RAG#"):
            from LaSSI.similarities.RAG import rag
            matrices = rag(self.legacy_conf.HuggingFace, self.catabolites_dir, obj_list)
        elif self.transformation == SentenceRepresentation.FullText and self.legacy_conf.HuggingFace.startswith("Log#"):
            f = Classifier(self.legacy_conf.HuggingFace[4:])
            matrices = []
            for i, x in enumerate(obj_list):
                ls = []
                for j, y in enumerate(obj_list):
                    ls.append(f(x, y))
                matrices.append(ls)
        else:
            if self.transformation == SentenceRepresentation.FullText:
                f = self.fulltext_similarity
            if self.transformation == SentenceRepresentation.Logical:
                # from LaSSI.Parmenides.TBox.CrossMatch import DoExpand  # LogicalGraph
                # doexp = DoExpand()
                # f = SentenceExpansion(obj_list, doexp, self.catabolites_of_dataset)

                self.logger("Starting the TBox Reasoning service")
                from LaSSI.structures.extended_fol.TBoxReasoning import TBoxReasoningSingleton
                TBoxReasoningSingleton.instance()
                # TODO: move the txt files to the resources
                if not os.path.exists(os.path.join(self.catabolites_of_dataset, str(self.full_transformation))):
                    from pathlib import Path
                    Path(os.path.join(self.catabolites_of_dataset, str(self.full_transformation))).mkdir(parents=True, exist_ok=True)
                kexp_pickle = os.path.join(self.catabolites_of_dataset, str(self.full_transformation), "_kexp.pickle")
                TBoxReasoningSingleton.init("query_impl.txt",
                                            "query_eq.txt",
                                            kexp_pickle)

                from LaSSI.structures.extended_fol.TabularCWASemantics import TabularCWASemantics
                f = TabularCWASemantics(obj_list, os.path.join(self.catabolites_of_dataset, str(self.full_transformation)))
                TBoxReasoningSingleton.instance().dump()
                # f.buildReport("rport")
            elif (self.transformation == SentenceRepresentation.LogicalGraph or
                  self.transformation == SentenceRepresentation.SimpleGraph):
                f = self.graph_with_logic_similarity

            matrix = None
            if self.matrix_file is not None:
                with open(self.matrix_file, "r") as ww:
                    matrix = json.load(ww)
            matrices = []

            # inv_it = list(reversed(list(enumerate(obj_list))))
            for i, x in enumerate(obj_list): #inv_it: #enumerate(obj_list):
                start = time.time()

                ls = []
                for j, y in enumerate(obj_list): #inv_it: #enumerate(obj_list):
                    eval = f(x, y)
                    if matrix is not None:
                        returned = matrix[i][j]
                        if (returned == 0.0 or returned == 1.0) and (returned == eval):
                            print(f"OK: {i} {j} with {eval} (expected: {returned})")
                        elif (eval != 0.0) and (eval != 1.0) and (returned == None):
                            print(f"OK: {i} {j} with {eval} (expected: {returned})")
                        else:
                            print(f"ERROR: {i} {j} with {eval} != {returned}")
                            f(x, y)
                    ls.append(eval)
                matrices.append(ls)

                end = time.time()
                self.sentences_benchmark.add_row(i, "Performing ex post explanation", end - start)
            # matrices = np.array(matrices)

        return matrices

    def ex_post_explain(self, lists):
        from LaSSI.files.FileDumpUtilities import target_file_dump
        self.logger("computing similarities")
        experiment_name = self.full_transformation.name + (f"_{self.legacy_conf.HuggingFace.split('/')[-1]}" if self.transformation == SentenceRepresentation.FullText else "")
        confusion_matrices = target_file_dump(self.confusion_matrices + experiment_name + ".json",
                                              json.load,
                                              lambda: CalculateMatrix(self, lists),
                                              json_dumps,
                                              self.force)

        if self.clusters_file is not None:
            from LaSSI.similarities.ClusteringTest import test_with_maximal_matching
            clusters = []
            with open(self.clusters_file, "r") as f:
                clusters = json.load(f)
            matrix = None
            if self.matrix_file is not None:
                with open(self.matrix_file, "r") as f:
                    matrix = json.load(f)
            test_with_maximal_matching(clusters, self.catabolites_dir,
                                       experiment_name, confusion_matrices, implication_matrix=matrix)

    def sentence_transform(self, sentences):
        if self.transformation == SentenceRepresentation.FullText:
            return sentences

        from LaSSI.files.FileDumpUtilities import target_file_dump
        n = len(sentences)
        self.logger("generating meuDB")
        if self.disable_a_priori:
            self.meu_dbs = ExplainTextWithNER(self, sentences)
            meu_execution_time = [0.0, 'r']
        else:
            self.meu_dbs, meu_execution_time = target_file_dump(
                self.meuDB,
                lambda x: [MeuDB.from_dict(k) for k in json.load(x)],
                lambda: ExplainTextWithNER(self, sentences),
                json_dumps, self.force, self.should_benchmark
            )
        self.logger(f"Generating meuDB time: {meu_execution_time} seconds")

        self.logger("generating gsmDB")
        gsm_db, gsm_execution_time = target_file_dump(
            self.gsmDB,
            lambda x: x.read(),
            lambda: GetGSMString(self, sentences),
            lambda x: x,
            self.force, self.should_benchmark
        )
        self.logger(f"Generating gsmDB time: {gsm_execution_time} seconds")

        self.logger("generating rewritten graphs")
        rewritten_graphs, rewritten_execution_time = target_file_dump(
            self.datagramdb_output,
            json.load,
            lambda: ApplyGraphGrammars(self, n),
            json_dumps, self.force, self.should_benchmark
        )
        print(f"Generating rewritten graphs time: {rewritten_execution_time} seconds")

        self.logger("generating intermediate representation (before final logical form in eFOL)")
        is_binary = False
        intermediate_representations, intermediate_execution_time = target_file_dump(
            self.internals,
            obj_unmarshall if is_binary else lambda x: [InternalRepresentation.from_dict(k) for k in json.load(x)],
            lambda: SemanticGraphRewriting(self, rewritten_graphs),
            obj_pickle if is_binary else json_dumps, not is_binary, self.should_benchmark, is_binary
        )
        print(f"Generating intermediate representations time: {intermediate_execution_time} seconds")
        write_variable_to_file(self.benchmarking_file,
                                    f"{self.get_execution_time_string(meu_execution_time)},{gsm_execution_time[0]},{rewritten_execution_time[0]},{intermediate_execution_time[0]},")
        if self.transformation == SentenceRepresentation.Logical:  # LogicalGraph
            logical_representations, logical_rewriting_execution_time = target_file_dump(
                self.logical_rewriting,
                lambda x: formula_from_dict(json.load(x)),
                lambda: LogicalRewriting(self, intermediate_representations),
                json_dumps, self.force, self.should_benchmark)
            write_variable_to_file(self.benchmarking_file,f"{logical_rewriting_execution_time[0]}")
            print(f"Generating logical representations time: {logical_rewriting_execution_time} seconds")
        else:
            logical_representations = intermediate_representations
            write_variable_to_file(self.benchmarking_file, f"{None}")
        return logical_representations

    def get_execution_time_string(self, execution_time):
        if 'w' == execution_time[1]:
            return f"{execution_time[0]},0"
        elif 'r' == execution_time[1]:
            return f"0,{execution_time[0]}"
        return None

    def run(self):
        from LaSSI.phases.SentenceLoader import SentenceLoader

        start_time = time.time()
        sentences = SentenceLoader(self.sentences)
        end_time = time.time()
        loading_sentences_execution_time = end_time - start_time
        self.logger(f"Loading sentences time: {loading_sentences_execution_time} seconds")
        write_variable_to_file(self.benchmarking_file,
                                    f"{self.dataset_name.split('/')[-1].split('.yaml')[0]},{loading_sentences_execution_time},")

        result = self.sentence_transform(sentences)

        # TODO: Still not working for 200 sentences, will this be fixed by next week (or still ignoring for this paper)?
        if self.run_ex_post:
            start_time = time.time()
            self.ex_post_explain(result)
            end_time = time.time()
            ex_post_execution_time = end_time - start_time
            self.logger(f"Ex Post Time: {loading_sentences_execution_time} seconds")
            write_variable_to_file(self.benchmarking_file,
                                        f",{ex_post_execution_time}\n")
        else:
            write_variable_to_file(self.benchmarking_file,
                                        f"\n")

        self.sentences_benchmark.to_csv()


    def close(self):
        if isinstance(self.sentences, io.IOBase):
            self.sentences.close()
            self.logger("~~DONE~~")
