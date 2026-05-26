import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class NLIClassifier:
    def __init__(self, model_name="BAAI/bge-reranker-v2-m3"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def __call__(self, premise: str, consequence: str) -> float:
        pairs = [[premise, consequence]]
        with torch.no_grad():
            inputs = self.tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512).to(self.device)
            scores = self.model(**inputs, return_dict=True).logits.squeeze().float()
            
            if scores.numel() == 1:
                # BGE reranker outputs logits, use sigmoid to normalize to [0, 1]
                score = torch.sigmoid(scores).item()
            else:
                probs = torch.softmax(scores, dim=-1)
                labels = self.model.config.id2label
                
                target_idx = None
                for idx, label in labels.items():
                    label_lower = str(label).lower()
                    if 'entailment' in label_lower or 'positive' in label_lower or 'label_2' in label_lower:
                        target_idx = int(idx)
                        break
                
                if target_idx is not None:
                    score = probs[target_idx].item()
                else:
                    # Fallback to last index (often positive/entailment in default configs)
                    score = probs[-1].item()
                    
        return score
