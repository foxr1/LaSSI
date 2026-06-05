import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class NLIClassifier:
    def __init__(self, model_name="BAAI/bge-reranker-v2-m3"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()

    # Strict label → float mapping for 3-class NLI models
    _LABEL_SCORES = {
        "entailment": 1.0,
        "neutral":    0.5,
        "contradiction": 0.0,
        # common aliases
        "positive":   1.0,
        "negative":   0.0,
    }

    def _label_to_score(self, label_str: str) -> float:
        """Map a predicted NLI label string to a crisp float."""
        key = label_str.lower().strip()
        for name, val in self._LABEL_SCORES.items():
            if name in key:
                return val
        # Fallback for opaque label names like 'LABEL_2': treat highest index as entailment
        return 0.5

    def __call__(self, premise: str, consequence: str) -> float:
        pairs = [[premise, consequence]]
        with torch.no_grad():
            inputs = self.tokenizer(pairs, padding=True, truncation=True,
                                    return_tensors='pt', max_length=512).to(self.device)
            logits = self.model(**inputs, return_dict=True).logits.squeeze().float()

            if logits.numel() == 1:
                # Binary re-ranker (e.g. BGE): single relevance logit → threshold to 0 / 1
                score = 1.0 if torch.sigmoid(logits).item() >= 0.5 else 0.0
            else:
                # 3-class NLI: take the argmax label and map to a strict value
                pred_idx = int(logits.argmax().item())
                pred_label = str(self.model.config.id2label[pred_idx])
                score = self._label_to_score(pred_label)

        return score
