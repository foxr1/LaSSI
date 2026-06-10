import os
import shutil
from pathlib import Path

def delete_files(delete_all_files=False, benchmarking=False, target_folders=None, transformation=None):
    if target_folders is None:
        target_folders = []
    catabolites_dir = os.path.join(Path(os.path.dirname(os.path.abspath(__file__))).parent.absolute().parent.absolute(), "catabolites")
    for subdir, dirs, files in os.walk(catabolites_dir):
        if (subdir.split(os.sep)[-1][0].isdigit() and benchmarking) or (subdir.split(os.sep)[-1] in target_folders and not benchmarking):
            for in_dir in dirs:
                if (in_dir in ("viz", "matrices")) or (transformation is not None and in_dir == f"SentenceRepresentation.{transformation.name}"):
                    dir_path = os.path.join(subdir, in_dir)
                    try:
                        print(f"Deleting folder: {str(dir_path)}")
                        try:
                            shutil.rmtree(dir_path)
                        except OSError as e:
                            print(f"Error deleting {dir_path}: {e}")
                    except ValueError as e:
                        print(e)
            for file in files:
                if (
                        file in ("gsmDB.txt", "datagramdb_output.json", "logical_rewriting.json")
                        or file.endswith(".pickle") or file.startswith("confusion_matrices")
                        or (file in ("internals.json", "internals-bin.json", "string_rep.txt", "meuDBs.json") and delete_all_files)
                        or (
                            (file in ("internals.json", "logical_rewriting.json", "gsmDB.txt", "datagramdb_output.json")
                                or file.startswith("confusion_matrices"))
                            and len(target_folders) > 0
                        )
                ):
                    file_path = os.path.join(subdir, file)
                    print(f"Deleting file: {str(file_path)}")
                    try:
                        os.remove(file_path)
                    except OSError as e:
                        print(f"Error deleting {file_path}: {e}")

if __name__ == '__main__':
    delete_files()
