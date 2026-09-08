"""Pinned English TweetEval model; deterministic CPU inference."""
import numpy as np

from .collect import digest, immutable, json_bytes

MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
REVISION = "3216a57f2a0d9c45a2e6c20157c20c49fb4bf9c7"
LABELS = ["negative", "neutral", "positive"]


def preprocess(text):
    return " ".join("@user" if t.startswith("@") and len(t) > 1 else
                    "http" if t.startswith("http") else t for t in text.split())


def validate(probabilities):
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any() or not np.allclose(p.sum(axis=1), 1, atol=1e-6):
        raise ValueError("Invalid negative/neutral/positive probabilities")
    return p


class SentimentModel:
    def __init__(self, download=False):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        self.torch = torch
        kwargs = {"revision": REVISION, "local_files_only": not download, "trust_remote_code": False}
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL, **kwargs)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL, weights_only=True, **kwargs).cpu().eval()
        if [self.model.config.id2label[i].lower() for i in range(3)] != LABELS:
            raise ValueError("Unexpected model label order")

    def predict(self, texts):
        outputs = []
        for start in range(0, len(texts), 16):
            tokens = self.tokenizer([preprocess(t) for t in texts[start:start + 16]],
                                    return_tensors="pt", padding=True, truncation=True, max_length=512)
            with self.torch.inference_mode():
                outputs.extend(self.model(**tokens).logits.softmax(dim=-1).numpy().tolist())
        return validate(outputs)


def score(frame, model):
    result = frame.copy()
    probabilities = model.predict(result.text.tolist()) if len(result) else np.empty((0, 3))
    probabilities = validate(probabilities)
    for index, label in enumerate(LABELS):
        result[f"prob_{label}"] = probabilities[:, index]
    result["sentiment_score"] = probabilities[:, 2] - probabilities[:, 0]
    result["sentiment_label"] = [LABELS[i] for i in probabilities.argmax(axis=1)]
    result["model"] = MODEL
    result["model_revision"] = REVISION
    return result


class CachedModel:
    """Keep a text's probabilities stable across later collection batches."""
    def __init__(self, folder, factory):
        self.folder, self.factory = folder, factory

    def predict(self, texts):
        import json
        values, missing = {}, {}
        for text in texts:
            key = digest(text.encode())
            path = self.folder / f"{key}.json"
            if path.exists():
                record = json.loads(path.read_bytes())
                if record["text_sha256"] != key or record["model_revision"] != REVISION:
                    raise ValueError("Scored cache identity mismatch")
                values[key] = validate([record["probabilities"]])[0].tolist()
            else:
                missing[key] = text
        if missing:
            keys = sorted(missing)
            predictions = self.factory().predict([missing[k] for k in keys])
            for key, probabilities in zip(keys, validate(predictions)):
                values[key] = probabilities.tolist()
                immutable(self.folder / f"{key}.json", json_bytes({"text_sha256": key,
                    "model_revision": REVISION, "probabilities": values[key]}))
        return validate([values[digest(text.encode())] for text in texts])
