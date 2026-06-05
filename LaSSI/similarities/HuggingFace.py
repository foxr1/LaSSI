import numpy as np
from sentence_transformers import SentenceTransformer, util

class HuggingFace:
    def __init__(self, model=None):
        if model is None:
            model = 'sentence-transformers/all-MiniLM-L6-v2'
        self.model = SentenceTransformer(model)

    def getEmbedding(self, x):
        return self.model.encode(x)

    def string_similarity(self, x: str, y: str) -> float:  # between 0 and 1
        x_vec = self.model.encode(x)
        y_vec = self.model.encode(y)
        strictSim = float(util.pairwise_dot_score(x_vec, y_vec))
        if (strictSim > np.finfo(float).eps): #ReLU
            return strictSim
        return 0.0

    # Fixed-baseline thresholds for mapping sentence-transformer similarity scores
    # onto {refuted, neutral, supported}. Chosen a priori (not tuned on the test
    # set) so the baseline stays methodologically defensible; sentence similarity
    # is a symmetric, non-directional signal, so over-tuning these is misleading.
    TAU_HIGH = 0.8
    TAU_LOW = 0.2

    @classmethod
    def label(cls, score: float) -> str:
        if score >= cls.TAU_HIGH:
            return "supported"
        if score <= cls.TAU_LOW:
            return "refuted"
        return "neutral"