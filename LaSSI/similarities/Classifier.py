# https://aclanthology.org/2024.findings-acl.353/
def _best_device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Classifier:
    def __init__(self, model=None):
        if model is None:
            model = "qbao775/AMR-LE-DeBERTa-V2-XXLarge-Contraposition-Double-Negation-Implication-Commutative-Pos-Neg-1-3"
        from transformers import pipeline
        self.pipe = pipeline("text-classification", model, device=_best_device())

    def __call__(self, premise, consequence):
        prompt = f"{premise}. {consequence}."
        result = self.pipe(prompt)[0]
        if (result["label"] == 1):
            score =  result["score"]/2.0+0.5
        elif (result["label"] == 0):
            score = (1-result["score"])/2.0
        else:
            raise RuntimeError("ERROR: unexpected label {}".format(result["label"]))
        print(f"Prompt: '{prompt}'. Score: {result['score']}. Label: {result['label']}")
        return score


if __name__ == "__main__":
    pipe = Classifier()
    print(pipe("Alice skates", "Bob skates"))             # Inferred Indifference? Class 0, Score 0.99 -> >0.01 score after normalization ~ Still, this merges with conflicting information
    print(pipe("Alice Skates", "Alice does skate"))       # Correct implication    Class 1, Score 0.77 -> 0.88 after normalization
    print(pipe("Alice Skates", "Alice does play sports")) # Inferred semantics?   Class 1, Score 0.76  -> 0.88 after normalization
    print(pipe("Alice skates","Alice does not skate"))    # Wrong classification! Class 1, Score 0.97  -> 0.98 after normalization
