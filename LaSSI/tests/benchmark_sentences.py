from LaSSI.Configuration import SentenceRepresentation
from LaSSI.tests.delete_catabolites import delete_files
from LaSSI.tests.run_all_sentences import get_and_run_all_sentences


def benchmark_sentences(iterations=1):
    for j in range(iterations):
        get_and_run_all_sentences(["evidence_cases"])
        if (j + 1) < iterations:
            delete_files(False, True, transformation=SentenceRepresentation.Logical)


if __name__ == '__main__':
    num_of_iterations = 1
    should_rerun_meuDB_generation = False

    if should_rerun_meuDB_generation:
        for i in range(num_of_iterations):
            delete_files(True, True, transformation=SentenceRepresentation.Logical)  # First delete ALL files (inc. meuDB) and only benchmarking files
            benchmark_sentences(num_of_iterations)
    else:
        delete_files(False, True, transformation=SentenceRepresentation.Logical)  # First delete ONLY benchmarking files
        benchmark_sentences(num_of_iterations)
