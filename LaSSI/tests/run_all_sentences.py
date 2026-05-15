import glob
import os
import re
import sys
import warnings
import multiprocessing
from multiprocessing import cpu_count
from pathlib import Path

from tqdm import tqdm

from LaSSI.Configuration import SentenceRepresentation
from LaSSI.LaSSI import LaSSI
from LaSSI.tests.benchmark import Benchmark
from LaSSI.tests.delete_catabolites import delete_files

# Prevent deadlocks in child processes
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# LaSSI/Transformers often spawn their own child processes. Daemonic pool workers
# cannot have children. This custom context creates non-daemonic workers.
_base_ctx = multiprocessing.get_context()


class NonDaemonProcess(_base_ctx.Process):
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass


class NonDaemonContext(type(_base_ctx)):
    Process = NonDaemonProcess


custom_mp_context = NonDaemonContext()

# Ignore warnings globally if needed for multiprocessing
warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only=False.*")


def sort_by_numeric_value(file_path):
    match = re.search(r'(\d+).yaml', file_path)
    return int(match.group(1).split('.yaml')[0]) if match else 0


def process_yaml_file(args):
    yaml_file, transformation, transformer, script_dir = args
    os.chdir(script_dir)

    try:
        with open(os.devnull, 'w') as devnull:
            # sys.stdout = devnull # silence stdout

            pipeline = LaSSI(yaml_file, "connection.yaml", transformation, transformer)
            pipeline.run()
            pipeline.close()
        # sys.stdout = sys.__stdout__
        return (yaml_file, True, None)
    except Exception as e:
        return (yaml_file, False, str(e))


def get_and_run_all_sentences(folders, transformation=SentenceRepresentation.Logical,
                              transformer='sentence-transformers/all-MiniLM-L6-v2'):
    print(f"TRANSFORMATION: {transformation}, TRANSFORMER: {transformer}")
    root_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent.absolute().parent.absolute()
    sentences_dir = os.path.join(root_dir, "test_sentences")
    main_script_path = os.path.join(root_dir, "main.py")
    script_dir = os.path.dirname(os.path.abspath(main_script_path))

    # Collect all YAML files
    yaml_files = []
    for folder_name in folders:
        folder_path = os.path.join(sentences_dir, folder_name)
        yaml_files.extend(glob.glob(os.path.join(folder_path, "*.yaml")))

    yaml_files.sort(key=sort_by_numeric_value)
    os.chdir(script_dir)

    mp_args = [(yaml_file, transformation, transformer, script_dir) for yaml_file in yaml_files]
    with custom_mp_context.Pool(processes=2) as pool:
        with tqdm(total=len(yaml_files), desc="Rewriting sentences") as pbar:
            for result in pool.imap_unordered(process_yaml_file, mp_args):
                yaml_file, success, error_msg = result
                pbar.set_description(f"Processing: {os.path.basename(yaml_file)}")

                if not success:
                    tqdm.write(f"\nError running LaSSI for: {yaml_file}\n{error_msg}")

                pbar.update(1)


if __name__ == '__main__':
    import multiprocessing

    multiprocessing.freeze_support()

    if len(sys.argv) > 1:
        folders = sys.argv[1:]
    else:
        folders = ["bbc"]

    all_outputs = True
    metrics_benchmark = Benchmark("Metrics")

    if all_outputs:
        transformations = [
            SentenceRepresentation.SimpleGraph,
            SentenceRepresentation.LogicalGraph,
            SentenceRepresentation.FullText,
            SentenceRepresentation.Logical,
            SentenceRepresentation.SimpleGraphDisabledAPriori,
            SentenceRepresentation.LogicalGraphDisabledAPriori,
            SentenceRepresentation.LogicalDisabledAPriori
        ]

        for transformation in transformations:
            delete_files(False, False, folders, transformation)
            if transformation == SentenceRepresentation.FullText:
                transformers = [
                    "all-MiniLM-L6-v2", "all-MiniLM-L12-v2", "all-mpnet-base-v2", "all-roberta-large-v1",
                    "Log#qbao775/AMR-LE-DeBERTa-V2-XXLarge-Contraposition-Double-Negation-Implication-Commutative-Pos-Neg-1-3",
                    "RAG#colbert-ir/colbertv2.0"
                ]
                for transformer in transformers:
                    get_and_run_all_sentences(
                        folders, transformation,
                        f"sentence-transformers/{transformer}" if "#" not in transformer else transformer
                    )
            else:
                get_and_run_all_sentences(folders, transformation)
    else:
        get_and_run_all_sentences(folders)

    metrics_benchmark.to_csv("bbc.csv")