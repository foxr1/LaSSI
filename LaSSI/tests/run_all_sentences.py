import csv
import glob
import os
import re
import sys
import warnings
import multiprocessing
from pathlib import Path

import requests

from tqdm import tqdm

from LaSSI.Configuration import SentenceRepresentation
from LaSSI.LaSSI import LaSSI
from LaSSI.tests.benchmark import Benchmark
from LaSSI.tests.delete_catabolites import delete_files

# Prevent deadlocks in child processes
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Allow MPS (Apple Silicon) to fall back to CPU for ops it doesn't support,
# rather than raising a hard error.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

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


def _ollama_is_available(base_url="http://localhost:11434", timeout=5) -> bool:
    try:
        r = requests.get(base_url, timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


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


_CSV_FIELDNAMES = ["id", "transformation", "transformer", "ex_post_explain_s"]
_CSV_OUT_DIR = "results/sentence_length"


def _csv_path(filename):
    os.makedirs(_CSV_OUT_DIR, exist_ok=True)
    return os.path.join(_CSV_OUT_DIR, filename)


def _init_csv(filename="evidence_cases.csv"):
    """Create (or overwrite) the CSV with just a header row — call once at run start."""
    out_path = _csv_path(filename)
    with open(out_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=_CSV_FIELDNAMES).writeheader()
    return out_path


def _append_ex_post_rows(rows, filename="evidence_cases.csv"):
    """Append rows to the CSV immediately so partial results survive a crash."""
    if not rows:
        return
    with open(_csv_path(filename), "a", newline="") as f:
        csv.DictWriter(f, fieldnames=_CSV_FIELDNAMES).writerows(rows)


def get_and_run_all_sentences(folders, transformation=SentenceRepresentation.Logical,
                              transformer='sentence-transformers/all-MiniLM-L6-v2',
                              ex_post_rows=None,
                              csv_filename=None,
                              processes=2):
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
    with custom_mp_context.Pool(processes=processes) as pool:
        with tqdm(total=len(yaml_files), desc="Rewriting sentences") as pbar:
            for result in pool.imap_unordered(process_yaml_file, mp_args):
                yaml_file, success, error_msg, bench_data, bench_phases = result
                pbar.set_description(f"Processing: {os.path.basename(yaml_file)}")

                if not success:
                    tqdm.write(f"\nError running LaSSI for: {yaml_file}\n{error_msg}")
                elif ex_post_rows is not None and bench_data:
                    yaml_name = os.path.splitext(os.path.basename(yaml_file))[0]
                    new_rows = []
                    for row in bench_data:
                        ex_post_time = row.get("Performing ex post explanation")
                        if ex_post_time is not None:
                            new_rows.append({
                                "id": f"{yaml_name}_{row['id']}",
                                "transformation": transformation.name,
                                "transformer": short_transformer,
                                "ex_post_explain_s": ex_post_time,
                            })
                    ex_post_rows.extend(new_rows)
                    if csv_filename is not None:
                        _append_ex_post_rows(new_rows, csv_filename)

                pbar.update(1)


def write_ex_post_csv(ex_post_rows, filename="evidence_cases.csv"):
    """Print a completion summary. Data is already on disk if csv_filename was passed to get_and_run_all_sentences."""
    if not ex_post_rows:
        print("No ex post data to export.")
        return
    print(f"Run complete: {len(ex_post_rows)} rows written to {_csv_path(filename)}")


if __name__ == '__main__':
    import multiprocessing

    multiprocessing.freeze_support()

    if len(sys.argv) > 1:
        folders = sys.argv[1:]
    else:
        folders = ["evidence_cases"]

    all_outputs = True
    num_processes = 1  # Set to 1 for accurate single-sentence latency benchmarking
    csv_filename = "evidence_cases.csv"
    ex_post_rows = []
    metrics_benchmark = Benchmark("Metrics")

    _init_csv(csv_filename)  # Start fresh; rows are flushed to disk after every YAML file

    if all_outputs:
        transformations = [
            SentenceRepresentation.FullText,
            SentenceRepresentation.Logical
        ]

        for transformation in transformations:
            delete_files(False, False, folders, transformation)
            if transformation == SentenceRepresentation.FullText:
                llm_transformers = [
                    "LLM#llama3.2:3b", "LLM#qwen3.5:4b", "LLM#gemma4:e2b"
                ]
                transformers = [
                    "all-MiniLM-L6-v2", "all-MiniLM-L12-v2", "all-mpnet-base-v2",
                    "NLI#BAAI/bge-reranker-v2-m3", "NLI#cross-encoder/nli-deberta-v3-base", "NLI#cross-encoder/nli-MiniLM2-L6-H768",
                ]
                ollama_available = _ollama_is_available()
                if ollama_available:
                    transformers += llm_transformers
                else:
                    print("[run_all_sentences] Ollama not reachable at localhost:11434 — skipping LLM transformers.")
                for transformer in transformers:
                    get_and_run_all_sentences(
                        folders, transformation,
                        f"sentence-transformers/{transformer}" if "#" not in transformer else transformer,
                        ex_post_rows,
                        csv_filename=csv_filename,
                        processes=num_processes
                    )
            else:
                get_and_run_all_sentences(
                    folders, transformation,
                    ex_post_rows=ex_post_rows,
                    csv_filename=csv_filename,
                    processes=num_processes,
                )
    else:
        get_and_run_all_sentences(folders, csv_filename=csv_filename, processes=num_processes)

    write_ex_post_csv(ex_post_rows, csv_filename)
    metrics_benchmark.to_csv("evidence_cases.csv")
