"""Model-backed LSTM inference with a transparent rule-based cold-start mode."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "classifier.pt"

KEYWORDS = {
    "billing": ("refund", "charged", "charge", "payment", "pay", "invoice", "receipt", "billing", "price", "discount", "subscription", "card", "coupon", "installment", "فاتورة", "دفع", "فلوس", "استرداد", "خصم"),
    "technical": ("error", "crash", "login", "log in", "password", "bug", "can't open", "wont load", "won't load", "video", "broken", "progress", "reset", "slow", "not working", "technical", "upload", "freezing", "freezes", "freeze", "quiz", "submit", "assignment", "مشكلة", "خطأ", "مش شغال", "فيديو", "تسجيل الدخول"),
    "booking": ("booking", "book", "reserve", "availability", "available", "session", "schedule", "reschedule", "appointment", "slot", "office hours", "mentor", "حجز", "موعد", "جلسة"),
    "general": ("prerequisite", "prerequisites", "certificate", "syllabus", "topics", "courses", "beginner", "contact", "accessibility", "privacy", "community", "how long", "duration", "update my email", "learning", "curriculum", "شهادة", "كورس", "دورة"),
}


def classifier_backend() -> str:
    return "LSTM model" if MODEL_PATH.exists() else "Rules fallback"


@lru_cache(maxsize=1)
def _load():
    import torch
    from src.dataset import encode
    from src.model import IntentClassifier

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Checkpoints are produced locally by src.train, never load untrusted files.
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    cfg = ckpt["config"]
    model = IntentClassifier(vocab_size=cfg["vocab_size"], embed_dim=cfg["embed_dim"],
        hidden_dim=cfg["hidden_dim"], num_classes=cfg["num_classes"],
        rnn_type=cfg["rnn_type"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt["vocab"], ckpt["label_map"], cfg["max_len"], device, encode


def _rules(text: str) -> tuple[str, float]:
    lowered = text.lower()
    scores = {}
    for label, words in KEYWORDS.items():
        scores[label] = sum(1 for w in words if re.search(r"(?<!\w)" + re.escape(w) + r"(?!\w)", lowered))
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return "general", 0.35
    runner_up = sorted(scores.values())[-2]
    confidence = 0.92 if runner_up == 0 and scores[best] > 1 else 0.83 if runner_up == 0 else 0.63
    return best, confidence


def classify(text: str) -> tuple[str, float]:
    """Return (intent, confidence), even before training or without torch."""
    if not text.strip():
        return "general", 0.0
    fallback_label, fallback_conf = _rules(text)
    # Exact domain terms often outperform a small synthetic model on unseen requests.
    # Leave the trained model in control where confident and not contradicted.
    if not MODEL_PATH.exists():
        return fallback_label, fallback_conf
    try:
        import torch
        import torch.nn.functional as F
        model, vocab, id2label, max_len, device, encode = _load()
        ids, length = encode(text, vocab, max_len)
        with torch.inference_mode():
            logits = model(torch.tensor([ids], dtype=torch.long, device=device),
                           torch.tensor([length], dtype=torch.long, device=device))
            confidence, pred = F.softmax(logits, dim=1)[0].max(dim=0)
        label, conf = str(id2label[int(pred.item())]), float(confidence.item())
        if fallback_conf >= 0.83 and label != fallback_label:
            return fallback_label, fallback_conf
        return label, conf
    except (RuntimeError, ValueError, KeyError, ImportError, OSError) as exc:
        import logging
        logging.getLogger(__name__).warning("Classifier fallback after %s", type(exc).__name__)
        return fallback_label, fallback_conf
