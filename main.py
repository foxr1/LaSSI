import multiprocessing
import os
import sys
import yaml

from LaSSI.Configuration import SentenceRepresentation
from LaSSI.LaSSI import LaSSI
from datasets import load_dataset


def get_dataset(filename="test_sentences/commonsense/dataset.yaml", number=5):
    if not os.path.exists(filename):
        try:
            questions = load_dataset("tau/commonsense_qa")['train']['question'][:number]
            with open(filename, 'w') as f:
                yaml.dump(questions, f, default_flow_style=False, width=float('inf'))

        except Exception as e:
            print(e)
            return None

    return filename


if __name__ == '__main__':
    commonsense_qa = False

    if commonsense_qa:
        dataset_name = get_dataset()
    else:
        dataset_name = "test_sentences/orig/newcastle_mdpi.yaml"

    fuzzyDBs = "connection.yaml"

    if len(sys.argv) > 1:
        dataset_name = sys.argv[1]
    if len(sys.argv) > 2:
        fuzzyDBs = sys.argv[2]

    # Changes:
    # 1. SentenceRepresentation now have DisabledAdHoc variants (except from the Embedders)
    # 2. New embedding system, RAG#colbert-ir/colbertv2.0. To compare other systems for question answering, which is the thing we are targeting, I provided references to ColBERTv2, which is not only using embedding based, but also with "RAG#colbert-ir/colbertv2.0"
    # 3. For exploiting the implication classifier, I used a very recent paper also avialable through HuggingFace: Log#"Log#qbao775/AMR-LE-DeBERTa-V2-XXLarge-Contraposition-Double-Negation-Implication-Commutative-Pos-Neg-1-3"
    pipeline = LaSSI(dataset_name, fuzzyDBs, SentenceRepresentation.Logical)
    pipeline.run()
    pipeline.close()
