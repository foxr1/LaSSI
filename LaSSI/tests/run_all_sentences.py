import glob
import os
import re
import sys
from pathlib import Path

from tqdm import tqdm

from LaSSI.Configuration import SentenceRepresentation
from LaSSI.LaSSI import LaSSI
from LaSSI.tests.benchmark import Benchmark
from LaSSI.tests.delete_catabolites import delete_files


def sort_by_numeric_value(file_path):
    match = re.search(r'(\d+).yaml', file_path)
    return int(match.group(1).split('.yaml')[0]) if match else 0


def run_lassi_instance(args):
    yaml_file, transformation, transformer = args
    try:
        import os
        from LaSSI.LaSSI import LaSSI
        with open(os.devnull, 'w') as devnull:
            # sys.stdout = devnull
            pipeline = LaSSI(yaml_file, "connection.yaml", transformation, transformer)
            pipeline.run()
            pipeline.close()
        return yaml_file, None
    except Exception as e:
        return yaml_file, e


def get_and_run_all_sentences(folders, transformation=SentenceRepresentation.Logical, transformer='sentence-transformers/all-MiniLM-L6-v2'):
    print(f"TRANSFORMATION: {transformation}, TRANSFORMER: {transformer}")
    root_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent.absolute().parent.absolute()
    sentences_dir = os.path.join(root_dir, "test_sentences")
    main_script_path = os.path.join(root_dir, "main.py")

    yaml_files = []
    for folder_name in folders:
        folder_path = os.path.join(sentences_dir, folder_name)
        yaml_files.extend(glob.glob(os.path.join(folder_path, "*.yaml")))

    yaml_files.sort(key=sort_by_numeric_value)
    os.chdir(os.path.dirname(os.path.abspath(main_script_path)))

    import concurrent.futures
    with tqdm(total=len(yaml_files), desc="Rewriting sentences") as pbar:
        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
            args_list = [(yaml_file, transformation, transformer) for yaml_file in yaml_files]
            futures = {executor.submit(run_lassi_instance, args): args for args in args_list}
            for future in concurrent.futures.as_completed(futures):
                yaml_file, error = future.result()
                if error:
                    print(f"\nError running LaSSI for: {yaml_file}", error, file=sys.stderr)
                pbar.update(1)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        folders = sys.argv[1:]
    else:
        folders = ["bbc"]

    all_outputs = True
    metrics_benchmark = Benchmark("Metrics")

    if all_outputs:
        # transformations = [SentenceRepresentation.FullText, SentenceRepresentation.SimpleGraph, SentenceRepresentation.LogicalGraph, SentenceRepresentation.Logical]
        # transformations = [SentenceRepresentation.FullText, SentenceRepresentation.SimpleGraphDisabledAdHoc, SentenceRepresentation.SimpleGraph, SentenceRepresentation.LogicalGraphDisabledAdHoc, SentenceRepresentation.LogicalGraph, SentenceRepresentation.LogicalDisabledAdHoc, SentenceRepresentation.Logical]
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
