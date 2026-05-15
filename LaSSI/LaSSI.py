__author__ = "Giacomo Bergami"
__copyright__ = "Copyright 2020, Giacomo Bergami"
__credits__ = ["Giacomo Bergami"]
__license__ = "GPL"
__version__ = "2.0"
__maintainer__ = "Giacomo Bergami"
__email__ = "bergamigiacomo@gmail.com"
__status__ = "Production"

import collections
import hashlib
import io
import json
import multiprocessing
import os.path
import re
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
                 disable_fuzzy_honk: bool = False,
                 run_ex_post: bool = True,
                 useId:bool = False,
                 use_multiprocessing=True,
                 generate_png_matrix=True,
                 ):
        self.use_multiprocessing = use_multiprocessing
        if use_multiprocessing:
            try:
                multiprocessing.set_start_method('spawn')
            except RuntimeError:
                pass
        self.useId = useId
        self.disable_a_priori = disable_a_priori
        self.disable_fuzzy_honk = disable_fuzzy_honk
        self._generate_png_matrix_param = generate_png_matrix
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

        if self._generate_png_matrix_param is not None:
            self.generate_png_matrix = self._generate_png_matrix_param
        elif hasattr(fuzzyDBs, "generate_png_matrix"):
            self.generate_png_matrix = fuzzyDBs.generate_png_matrix
        else:
            self.generate_png_matrix = False

        self.legacy_conf.generate_png_matrix = self.generate_png_matrix
        # if hasattr(fuzzyDBs, "huggingface") and fuzzyDBs.huggingface is not None:
        #     self.legacy_conf.HuggingFace = fuzzyDBs.huggingface

        self.logger("init non-postgres services and the wrapper for the former...")
        self.initServices = Services.getInstance(self.logger)

        if not self.disable_fuzzy_honk:
            self.logger("init postgres services...")
            self.logger(" - Initialising the connection to the database")
            (FuzzyStringMatchDatabase
             .instance()
             .init(fuzzyDBs.db, fuzzyDBs.uname, fuzzyDBs.pw, fuzzyDBs.host, fuzzyDBs.port))

            self.logger(" - Loading the tab files or streaming those remotely, if required.")
            import tempfile
            if fuzzyDBs.fuzzy_dbs is not None:
                for k, v in fuzzyDBs.fuzzy_dbs.items():
                    self.logger(f" - Loading {k}.")
                    FuzzyStringMatchDatabase.instance().create(k, v)
        else:
            self.logger("skipping postgres services initialization (disable_fuzzy_honk=True)")

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


        from LaSSI.HOnK.HOnK import HOnKSingleton
        HOnKSingleton.instance()
        # Only call init if not already initialized (main_remote.py does this)
        if HOnKSingleton.get() is None:
            HOnKSingleton.init("cache", fuzzyDBs.uname, fuzzyDBs.pw,
                                     fuzzyDBs.host, fuzzyDBs.port, False, "LaSSI/HOnK.ttl",
                                     rules_path="raw_data/logical_analysis.json")
        
        parmo = HOnKSingleton.get()
        self.initServices.setHOnK(parmo)
        
        # Check if we are in remote mode (e.g. ParmenidesRemote)
        is_remote = hasattr(parmo, "endpoint_url")
        
        if is_remote or self.disable_fuzzy_honk:
            if self.disable_fuzzy_honk:
                print("[LaSSI] disable_fuzzy_honk=True — skipping local TTL hash check and table rebuild.")
            else:
                print("[LaSSI] Remote ontology detected — skipping local TTL hash check and table rebuild.")
            return

        honk_ttl_path = "LaSSI/HOnK.ttl"
        if not os.path.exists(honk_ttl_path):
            print(f"[LaSSI] WARNING: {honk_ttl_path} not found. Skipping local table rebuild.")
            return

        honk_hash_path = os.path.join("cache", "honk_oxstore.mtime")

        current_honk_hash = str(os.path.getmtime(honk_ttl_path))
        stored_honk_hash = None
        if os.path.exists(honk_hash_path):
            with open(honk_hash_path, "r") as _hf:
                stored_honk_hash = _hf.read().strip()
        honk_changed = current_honk_hash != stored_honk_hash
        if honk_changed:
            with tempfile.NamedTemporaryFile() as honk_tab:
                with open(honk_tab.name, 'w') as f:
                    self.initServices.getHOnK().dumpTypedObjectsToTAB(f)
                FuzzyStringMatchDatabase.instance().create("honk", honk_tab.name, '(id integer NOT NULL, idx text, t text, type text)', force=True)
            os.makedirs("cache", exist_ok=True)
            with open(honk_hash_path, "w") as _hf:
                _hf.write(current_honk_hash)
        else:
            print("HOnK.ttl unchanged — skipping honk table rebuild")

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
                       full_server_output=False,
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
        import traceback
        internal_representations = []
        for idx, (graph, meu_db) in enumerate(zip(gsm_list, self.meu_dbs)):
            start = time.time()
            from LaSSI.structures.provenance.GraphProvenance import GraphProvenance
            g = GraphProvenance(graph, meu_db, self.transformation == SentenceRepresentation.SimpleGraph)
            self.logger(f"{meu_db.first_sentence}")
            write_variable_to_file(self.string_rep_dir, meu_db.first_sentence)
            try:
                internal_graph = g.internal_graph()
                final_form = internal_graph
                if self.transformation == SentenceRepresentation.Logical:
                    final_form = g.sentence()
                    write_variable_to_file(self.string_rep_dir, f" ⇒ {final_form.to_string()}\n")
            except Exception as e:
                self.logger(f"ERROR on sentence {idx}: {e}\n{traceback.format_exc()}")
                write_variable_to_file(self.string_rep_dir, f" ⇒ ERROR\n")
                final_form = None
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
            if x is None:
                rewritten_kernels.append(None)
            else:
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
                # from LaSSI.HOnK.TBox.CrossMatch import DoExpand  # LogicalGraph
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
                pairwise_path = os.path.join(self.catabolites_of_dataset, str(self.full_transformation), "pairwise_truth_tables.json")
                if not os.path.exists(pairwise_path):
                    from tqdm import tqdm as _tqdm_pw
                    n_sent = len(obj_list)
                    pairwise_out = []
                    total_pairs = n_sent * (n_sent - 1)
                    slowest = (-1.0, None, None)
                    pw_total_elapsed = 0.0
                    self.logger(f"Building pairwise truth tables ({total_pairs} pairs)...")
                    with _tqdm_pw(total=total_pairs, desc="Pairwise truth tables", unit="pair") as pw_pbar:
                        for si in range(n_sent):
                            for sj in range(n_sent):
                                if si == sj:
                                    continue
                                _t0 = time.time()
                                expl = f.get_explained_id_similarity(si, sj)
                                _elapsed = time.time() - _t0
                                pw_total_elapsed += _elapsed
                                if _elapsed > slowest[0]:
                                    slowest = (_elapsed, si, sj)
                                atoms = {}
                                atoms.update({str(k): str(v) for k, v in expl.lhs.atoms.items()})
                                atoms.update({str(k): str(v) for k, v in expl.rhs.atoms.items()})
                                table_df = expl.explained_joined_table_natural_joined_with_operands
                                pairwise_out.append({
                                    "i": si,
                                    "j": sj,
                                    "confidence": float(expl.confidence),
                                    "atoms": atoms,
                                    "result_col_i": f"R{si}",
                                    "result_col_j": f"R{sj}",
                                    "rows": table_df.to_dict(orient="records"),
                                })
                                pw_pbar.set_postfix({
                                    "last": f"({si},{sj})",
                                    "t": f"{_elapsed:.2f}s",
                                    "slowest": f"({slowest[1]},{slowest[2]})@{slowest[0]:.1f}s",
                                })
                                pw_pbar.update(1)
                    self.logger(
                        f"  Pairwise truth tables done: {total_pairs} pairs in {pw_total_elapsed:.1f}s "
                        f"(slowest ({slowest[1]},{slowest[2]}) @ {slowest[0]:.1f}s)"
                    )
                    with open(pairwise_path, "w") as _pw_f:
                        json.dump(pairwise_out, _pw_f)
            elif (self.transformation == SentenceRepresentation.LogicalGraph or
                  self.transformation == SentenceRepresentation.SimpleGraph):
                f = self.graph_with_logic_similarity

            matrix = None
            if self.matrix_file is not None:
                with open(self.matrix_file, "r") as ww:
                    matrix = json.load(ww)

            from tqdm import tqdm

            n = len(obj_list)
            matrices = [None] * n

            def _compute_row(args):
                row_i, row_x, cell_pbar = args
                t0 = time.time()
                row_vals = []
                for col_j, col_y in enumerate(obj_list):
                    cell_t0 = time.time()
                    val = f(row_x, col_y)
                    cell_elapsed = time.time() - cell_t0
                    if matrix is not None:
                        returned = matrix[row_i][col_j]
                        if (returned == 0.0 or returned == 1.0) and (returned == val):
                            print(f"OK: {row_i} {col_j} with {val} (expected: {returned})")
                        elif (val != 0.0) and (val != 1.0) and (returned is None):
                            print(f"OK: {row_i} {col_j} with {val} (expected: {returned})")
                        else:
                            print(f"ERROR: {row_i} {col_j} with {val} != {returned}")
                            f(row_x, col_y)
                    row_vals.append(val)
                    cell_pbar.update(1)
                    cell_pbar.set_postfix({"last_cell": f"{cell_elapsed:.1f}s", "row": row_i, "col": col_j})
                return row_i, row_vals, time.time() - t0

            from concurrent.futures import ThreadPoolExecutor, as_completed
            max_workers = min(n, os.cpu_count() or 4, 8)
            with tqdm(total=n * n, desc="Computing similarity matrix", unit="cell") as cell_pbar:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {executor.submit(_compute_row, (i, x, cell_pbar)): i
                               for i, x in enumerate(obj_list)}
                    for future in as_completed(futures):
                        i, row, elapsed = future.result()
                        matrices[i] = row
                        self.sentences_benchmark.add_row(i, "Performing ex post explanation", elapsed)
            # matrices = np.array(matrices)

        return matrices

    def _generate_confusion_matrix_png(self, matrix, experiment_name, labels=None):
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import numpy as np
            import textwrap

            data = np.array(matrix)
            
            if labels is None:
                labels = [str(i+1) for i in range(data.shape[0])]
            
            # Wrap labels for display
            display_labels = ['\n'.join(textwrap.wrap(str(l), width=40)) for l in labels]

            # Dynamically calculate figure size based on matrix size and label length
            # Estimate height needed for labels
            max_label_lines = max([l.count('\n') for l in display_labels]) + 1
            cell_size = 1.2
            fig_width = max(12, data.shape[1] * cell_size + 4)
            fig_height = max(10, data.shape[0] * cell_size + (max_label_lines * 0.2))

            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            im = ax.imshow(data, cmap='viridis')

            # Add colorbar
            cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)

            # Show all ticks and label them with the respective list entries
            ax.set_xticks(np.arange(data.shape[1]))
            ax.set_yticks(np.arange(data.shape[0]))
            ax.set_xticklabels(display_labels, rotation=45, ha="right", fontsize=9)
            ax.set_yticklabels(display_labels, fontsize=9)

            # Loop over data dimensions and create text annotations.
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    ax.text(j, i, f"{data[i, j]:.2f}",
                                   ha="center", va="center", color="w" if data[i,j] < 0.2 else "black", fontsize=8)

            ax.set_title(f"Confusion Matrix: {experiment_name}", fontsize=14, pad=20)
            
            # Adjust layout to make room for labels
            plt.tight_layout()

            png_path = self.confusion_matrices + experiment_name + ".png"
            plt.savefig(png_path, bbox_inches='tight', dpi=150)
            plt.close(fig)
            self.logger(f"Confusion matrix PNG saved to {png_path}")
        except Exception as e:
            self.logger(f"Failed to generate confusion matrix PNG: {e}")

    def ex_post_explain(self, lists):
        from LaSSI.files.FileDumpUtilities import target_file_dump
        self.logger("computing similarities")
        experiment_name = self.full_transformation.name + (f"_{self.legacy_conf.HuggingFace.split('/')[-1]}" if self.transformation == SentenceRepresentation.FullText else "")
        confusion_matrices = target_file_dump(self.confusion_matrices + experiment_name + ".json",
                                              json.load,
                                              lambda: CalculateMatrix(self, lists),
                                              json_dumps,
                                              self.force)

        if self.generate_png_matrix:
            if self.transformation == SentenceRepresentation.FullText:
                labels = list(lists)
            elif self.meu_dbs is not None:
                labels = [m.first_sentence for m in self.meu_dbs]
            else:
                labels = [str(i+1) for i in range(len(confusion_matrices))]
            self._generate_confusion_matrix_png(confusion_matrices, experiment_name, labels)

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
        self._invalidate_stale_cache(n)
        self.logger("generating meuDB")
        if self.disable_a_priori:
            self.meu_dbs = ExplainTextWithNER(self, sentences)
            meu_execution_time = [0.0, 'r']
        else:
            self.meu_dbs, meu_execution_time = target_file_dump(
                self.meuDB,
                lambda x: [MeuDB.from_dict(k) for k in json.load(x)],
                lambda: ExplainTextWithNER(self, sentences, disable_fuzzy_honk=self.disable_fuzzy_honk),
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

        if (hasattr(self, 'row_to_sub_indices') and
                any(len(s) > 1 for s in self.row_to_sub_indices) and
                len(intermediate_representations) != len(self.row_to_sub_indices)):
            intermediate_representations = self._merge_intermediate_per_row(intermediate_representations)
            self.meu_dbs = self._merge_meu_dbs_per_row(self.meu_dbs)
            self._rewrite_string_rep_from_intermediate(intermediate_representations)
            self._persist_merged_caches(intermediate_representations)
            self._collapse_per_row_benchmark()

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

    def _invalidate_stale_cache(self, expected_count):
        expected_row_count = len(self.row_to_sub_indices) if hasattr(self, 'row_to_sub_indices') else expected_count
        
        # Internals are persisted per YAML row, but the lower-level graph caches
        # stay per sub-sentence.  MeuDBs feed graph→kernel construction, so the
        # on-disk MeuDB cache must also remain per sub-sentence; otherwise a
        # later rerun zips sub-sentence graphs against merged row-level MeuDBs
        # and shifts every sentence after the first split row.
        core_caches_valid = True
        for core_cache, expected in (
                (self.meuDB, expected_count),
                (self.internals, expected_row_count),
        ):
            if not os.path.isfile(core_cache):
                core_caches_valid = False
                break
            try:
                with open(core_cache, 'r') as f:
                    data = json.load(f)
                count = len(data) if isinstance(data, list) else None
                if count != expected:
                    core_caches_valid = False
                    break
            except Exception:
                core_caches_valid = False
                break

        for path in (self.datagramdb_output, self.internals, self.logical_rewriting, self.meuDB):
            if not os.path.isfile(path):
                continue
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                count = len(data) if isinstance(data, list) else None
            except Exception:
                count = None
            
            valid_count = expected_count if path in (self.datagramdb_output, self.meuDB) else expected_row_count
            
            should_delete = (count is None or count != valid_count)
            if path in (self.meuDB, self.internals, self.logical_rewriting) and not core_caches_valid:
                should_delete = True
                
            if should_delete:
                try:
                    os.remove(path)
                    if path == self.meuDB and os.path.isfile(self.gsmDB):
                        os.remove(self.gsmDB)
                except OSError:
                    pass

    _HEADER_TEXT_RE = re.compile(r'^\s*[A-Za-z][A-Za-z\-]*\s*:\s*\S')

    def _is_header_text(self, sub_text):
        """Detect a 'Header: body' label-value sub-sentence (e.g. ``Outcome: under
        investigation``). The colon separator is the structural cue: the word
        before is a category label rather than a real subject."""
        return isinstance(sub_text, str) and bool(self._HEADER_TEXT_RE.match(sub_text))

    def _ontology_construct_keys(self):
        """Set of property keys (uppercased construct names) that the HOnK logical
        rewriting rules emit, e.g. SPACE, TIME, TIME_STATUS, CAUSATION. Used to
        decide which sub-sentence properties to promote onto the primary kernel."""
        from LaSSI.external_services.Services import Services
        cached = getattr(self, '_ontology_construct_keys_cache', None)
        if cached is not None:
            return cached
        keys = set()
        try:
            rules = Services.getInstance().getHOnK().getLogicalRewritingRules() or {}
        except Exception:
            rules = {}
        for rule in rules.values():
            if getattr(rule, 'logicalConstructName', None):
                keys.add(rule.logicalConstructName.upper())
            for name, _ in (getattr(rule, 'additional_classifications', None) or []):
                if name:
                    keys.add(name.upper())
        self._ontology_construct_keys_cache = keys
        return keys

    def _merge_intermediate_per_row(self, intermediate_representations):
        from collections import defaultdict
        from LaSSI.structures.internal_graph.EntityRelationship import Singleton

        construct_keys = self._ontology_construct_keys()
        merged = []
        for row_position, sub_indices in enumerate(self.row_to_sub_indices):
            if not sub_indices:
                continue
            if len(sub_indices) == 1:
                merged.append(intermediate_representations[sub_indices[0]])
                continue
            primary = intermediate_representations[sub_indices[0]]
            if not isinstance(primary, Singleton):
                merged.append(primary)
                continue
            new_props = defaultdict(list)
            for k, v in dict(primary.properties).items():
                if isinstance(v, (list, tuple)):
                    new_props[k] = list(v)
                else:
                    new_props[k] = v
            for sub_idx in sub_indices[1:]:
                sub_kernel = intermediate_representations[sub_idx]
                if sub_kernel is None:
                    continue
                sub_text = self.meu_dbs[sub_idx].first_sentence if sub_idx < len(self.meu_dbs) else ''
                if (self._is_header_text(sub_text) and
                        isinstance(sub_kernel, Singleton)):
                    for k, v in dict(sub_kernel.properties).items():
                        if k not in construct_keys:
                            continue
                        items = list(v) if isinstance(v, (list, tuple)) else [v]
                        existing = new_props.get(k)
                        if existing is None:
                            new_props[k] = items
                        elif isinstance(existing, list):
                            new_props[k] = existing + items
                        else:
                            new_props[k] = [existing] + items
                    # new_props['SENTENCE'].append(sub_kernel)
                else:
                    new_props['SENTENCE'].append(sub_kernel)
            merged.append(primary.update_node_props(new_props))
        return merged

    def _merge_meu_dbs_per_row(self, meu_dbs):
        from LaSSI.structures.meuDB.meuDB import MeuDB

        merged = []
        for sub_indices in self.row_to_sub_indices:
            if not sub_indices:
                continue
            if len(sub_indices) == 1:
                merged.append(meu_dbs[sub_indices[0]])
                continue
            combined_first = " ".join(meu_dbs[i].first_sentence for i in sub_indices)
            combined_meu = []
            for i in sub_indices:
                combined_meu.extend(meu_dbs[i].multi_entity_unit)
            merged.append(MeuDB(first_sentence=combined_first, multi_entity_unit=combined_meu))
        return merged

    def _persist_merged_caches(self, intermediate_representations):
        """Overwrite the internals cache with the merged-per-row representation.

        Do not overwrite ``meuDBs.json`` here: graph rewriting caches are
        per-sub-sentence, and future reruns need a matching per-sub-sentence
        MeuDB cache while reconstructing internals.
        """
        try:
            with open(self.internals, 'w') as f:
                f.write(json_dumps(intermediate_representations))
        except Exception as exc:
            self.logger(f"Could not persist merged internals.json: {exc}")

    def _collapse_per_row_benchmark(self):
        """Aggregate per-sub-sentence rows in the sentences benchmark down to one
        row per YAML input row (sum the timings, max the sentence length).
        Otherwise the benchmark CSV reports more sentences than the user wrote."""
        if not getattr(self, 'sentences_benchmark', None):
            return
        data = self.sentences_benchmark.data
        if not data:
            return
        sub_to_row = {}
        for row_idx, sub_indices in enumerate(self.row_to_sub_indices):
            for sub_idx in sub_indices:
                sub_to_row[sub_idx] = row_idx
        merged_rows = {}
        for entry in data:
            sub_idx = entry.get('id')
            row_idx = sub_to_row.get(sub_idx, sub_idx)
            target = merged_rows.setdefault(row_idx, {'id': row_idx})
            for key, value in entry.items():
                if key == 'id':
                    continue
                if not isinstance(value, (int, float)):
                    target[key] = value
                    continue
                if key == 'Sentence length':
                    target[key] = max(target.get(key, 0), value)
                else:
                    target[key] = target.get(key, 0.0) + value
        data.clear()
        for row_idx in sorted(merged_rows):
            data.append(merged_rows[row_idx])

    def _rewrite_string_rep_from_intermediate(self, intermediate_representations):
        if self.string_rep_dir is None:
            return
        with open(self.string_rep_dir, 'w') as f:
            for idx, intermediate in enumerate(intermediate_representations):
                first_sentence = self.meu_dbs[idx].first_sentence if idx < len(self.meu_dbs) else ""
                try:
                    if intermediate is None:
                        f.write(f"{first_sentence} ⇒ ERROR\n")
                    elif self.transformation == SentenceRepresentation.Logical:
                        f.write(f"{first_sentence} ⇒ {intermediate.to_string()}\n")
                    else:
                        f.write(f"{first_sentence} ⇒ {intermediate}\n")
                except Exception:
                    f.write(f"{first_sentence} ⇒ ERROR\n")

    def get_execution_time_string(self, execution_time):
        if 'w' == execution_time[1]:
            return f"{execution_time[0]},0"
        elif 'r' == execution_time[1]:
            return f"0,{execution_time[0]}"
        return None

    def run(self):
        from LaSSI.phases.SentenceLoader import SentenceLoader, split_yaml_row_sentences

        start_time = time.time()
        raw_sentences = SentenceLoader(self.sentences)

        expanded = []
        self.row_to_sub_indices = []
        for row_text in raw_sentences:
            sub_sentences = split_yaml_row_sentences(str(row_text))
            sub_indices = []
            for sub in sub_sentences:
                sub_indices.append(len(expanded))
                expanded.append(sub)
            self.row_to_sub_indices.append(sub_indices)

        honk = self.initServices.getHOnK()
        if honk is not None:
            # Build from ontology; supplement with articles and coordinating conjunctions
            # that aren't modelled as Preposition/Conjunction in HOnK
            _honk_closed = (
                {p.lower() for p in honk.getPrepositions()} |
                {c.lower() for c in honk.getConjunctions()} |
                {'the', 'an', 'and', 'or', 'nor', 'yet', 'so'}
            )
        else:
            _honk_closed = None
        sentences = [self.normalise_text(s, _honk_closed) for s in expanded]

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
            self.logger(f"Ex Post Time: {ex_post_execution_time} seconds")
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

    # Fallback used when HOnK is unavailable at normalise_text call time
    _FALLBACK_CLOSED_CLASS = {
        "on", "in", "at", "by", "for", "with", "about", "against", "between",
        "into", "through", "during", "before", "after", "above", "below", "to",
        "from", "up", "down", "of", "off", "over", "under", "near", "and", "but",
        "or", "nor", "yet", "so", "the", "an",
    }

    def normalise_text(self, text, closed_class_lower=None):
        """
        Sanitizes input text to prevent StanfordNLP from misidentifying capitalized
        prepositions/conjunctions as proper nouns or roots.
        Also collapses consecutive duplicate closed-class words that arise from
        scraping artefacts (e.g. "recorded on On Or Near" → "recorded on or near").

        closed_class_lower: precomputed lowercase set from HOnK (prepositions +
        conjunctions). Falls back to _FALLBACK_CLOSED_CLASS when None.
        """
        if not isinstance(text, str):
            return text

        closed = closed_class_lower if closed_class_lower is not None else self._FALLBACK_CLOSED_CLASS

        words = text.split()
        if not words:
            return text

        sanitized_words = [words[0]]  # Preserve the capitalization of the first word

        for word in words[1:]:
            clean_word = "".join(c for c in word if c.isalpha()).lower()
            prev_clean = "".join(c for c in sanitized_words[-1] if c.isalpha()).lower()
            # A single uppercase letter immediately after a capitalised non-closed-class
            # word is a compound identifier (e.g. "Class A", "Type B") — preserve its
            # case so CoreNLP does not reparse it as a plain article/determiner.
            is_compound_id = (
                len(word) == 1 and word[0].isupper()
                and sanitized_words[-1][0].isupper()
                and prev_clean not in closed
            )
            normalised = word.lower() if (clean_word in closed and not is_compound_id) else word
            # Drop consecutive duplicate closed-class tokens (scraping artefact)
            if clean_word in closed and clean_word == prev_clean:
                continue
            sanitized_words.append(normalised)

        return " ".join(sanitized_words)
