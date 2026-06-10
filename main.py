import multiprocessing
import os
import sys
import yaml
import warnings

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only=False.*")

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
        dataset_name = "neet/evidence_cases/roadworks_002.yaml"

    fuzzyDBs = "connection.yaml"

    if len(sys.argv) > 1:
        dataset_name = sys.argv[1]
    if len(sys.argv) > 2:
        fuzzyDBs = sys.argv[2]

    # Changes:
    # 1. SentenceRepresentation now have DisabledAdHoc variants (except from the Embedders)
    # 2. New embedding system, RAG#colbert-ir/colbertv2.0. To compare other systems for question answering, which is the thing we are targeting, I provided references to ColBERTv2, which is not only using embedding based, but also with "RAG#colbert-ir/colbertv2.0"
    # 3. For exploiting the implication classifier, I used a very recent paper also avialable through HuggingFace: Log#"Log#qbao775/AMR-LE-DeBERTa-V2-XXLarge-Contraposition-Double-Negation-Implication-Commutative-Pos-Neg-1-3"
    # 4. Plain local LLM entailment (Ollama): transformer='LLM#qwen2.5' with SentenceRepresentation.FullText
    # 5. Ontology-grounded LLM (inject relations into the prompt, "Follow the Path" arXiv:2505.11140):
    #    transformer='LLMHOnK#qwen2.5' with SentenceRepresentation.FullText — compare against LLM#qwen2.5 to show grounding's effect.
    #    Sources are ablatable via 'LLMHOnK#<model>#<sources>' (sources omitted => all three):
    #      'LLMHOnK#qwen2.5#honk'                       — HOnK ontology only
    #      'LLMHOnK#qwen2.5#honk+lifecycle'             — + LifecycleStates.ttl contradictions
    #      'LLMHOnK#qwen2.5#honk+lifecycle+paraphrase'  — + Paraphrase.ttl equivalences (== bare 'LLMHOnK#qwen2.5')
    pipeline = LaSSI(dataset_name, fuzzyDBs, SentenceRepresentation.Logical, useId=True, use_multiprocessing=True, transformer='all-MiniLM-L6-v2')
    pipeline.run()
    pipeline.close()
