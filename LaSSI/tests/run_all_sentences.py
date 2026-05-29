import csv
import glob
import os
import re
import sys
import warnings
import multiprocessing
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
            bench_data = list(pipeline.sentences_benchmark.data)
            bench_phases = list(pipeline.sentences_benchmark.phase_names)
            pipeline.close()
        # sys.stdout = sys.__stdout__
        return (yaml_file, True, None, bench_data, bench_phases)
    except Exception as e:
        return (yaml_file, False, str(e), [], [])


def get_and_run_all_sentences(folders, transformation=SentenceRepresentation.Logical,
                              transformer='sentence-transformers/all-MiniLM-L6-v2',
                              ex_post_rows=None):
    print(f"TRANSFORMATION: {transformation}, TRANSFORMER: {transformer}")
    root_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent.absolute().parent.absolute()
    sentences_dir = os.path.join(root_dir, "test_sentences")
    main_script_path = os.path.join(root_dir, "main.py")
    script_dir = os.path.dirname(os.path.abspath(main_script_path))

    # For Logical, there is no meaningful transformer choice; record as null
    short_transformer = None if transformation == SentenceRepresentation.Logical else (
        transformer.split("/")[-1] if "/" in transformer else transformer
    )

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
                yaml_file, success, error_msg, bench_data, bench_phases = result
                pbar.set_description(f"Processing: {os.path.basename(yaml_file)}")

                if not success:
                    tqdm.write(f"\nError running LaSSI for: {yaml_file}\n{error_msg}")
                elif ex_post_rows is not None and bench_data:
                    yaml_name = os.path.splitext(os.path.basename(yaml_file))[0]
                    for row in bench_data:
                        ex_post_time = row.get("Performing ex post explanation")
                        if ex_post_time is not None:
                            ex_post_rows.append({
                                "id": f"{yaml_name}_{row['id']}",
                                "transformation": transformation.name,
                                "transformer": short_transformer,
                                "ex_post_explain_s": ex_post_time,
                            })

                pbar.update(1)


def write_ex_post_csv(ex_post_rows, filename="evidence_cases.csv"):
    if not ex_post_rows:
        print("No ex post data to export.")
        return

    out_dir = "results/sentence_length"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)

    fieldnames = ["id", "transformation", "transformer", "ex_post_explain_s"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ex_post_rows)

    print(f"Data successfully written to {out_path} ({len(ex_post_rows)} rows)")


if __name__ == '__main__':
    import multiprocessing

    multiprocessing.freeze_support()

    if len(sys.argv) > 1:
        folders = sys.argv[1:]
    else:
        folders = ["transformer_test"]

    all_outputs = True
    ex_post_rows = []
    metrics_benchmark = Benchmark("Metrics")

    if all_outputs:
        transformations = [
            SentenceRepresentation.FullText,
            SentenceRepresentation.Logical
        ]

        for transformation in transformations:
            delete_files(False, False, folders, transformation)
            if transformation == SentenceRepresentation.FullText:
                transformers = [
                    "all-MiniLM-L6-v2", "all-MiniLM-L12-v2", "all-mpnet-base-v2",
                    "NLI#BAAI/bge-reranker-v2-m3", "NLI#cross-encoder/nli-deberta-v3-base", "NLI#cross-encoder/nli-MiniLM2-L6-H768",
                    "LLM#llama3.1:latest", "LLM#qwen3.5:latest", "LLM#gemma4:e2b"
                ]
                for transformer in transformers:
                    get_and_run_all_sentences(
                        folders, transformation,
                        f"sentence-transformers/{transformer}" if "#" not in transformer else transformer,
                        ex_post_rows
                    )
            else:
                get_and_run_all_sentences(folders, transformation, ex_post_rows=ex_post_rows)
    else:
        get_and_run_all_sentences(folders)

    write_ex_post_csv(ex_post_rows)
    metrics_benchmark.to_csv("evidence_cases.csv")
