import math
import pickle
import re
from dataclasses import dataclass
from functools import lru_cache

from .config import LABEL_NAMES, MODEL_PATH


MIN_WORDS = 5
MIN_CONFIDENCE = 0.62
MIN_REAL_CONFIDENCE = 0.74
MIN_VOCAB_COVERAGE = 0.08

SUSPICIOUS_PATTERNS = [
    r"\bshocking\b",
    r"\bbreaking\b.*\b(secret|rumor|hoax|alien|conspiracy)\b",
    r"\bmiracle cure\b",
    r"\bcures?\s+cancer\b",
    r"\ball diseases\b",
    r"\bovernight\b",
    r"\binstantly\b",
    r"\bsecret\b",
    r"\bconspiracy\b",
    r"\bhoax\b",
    r"\brumou?r\b",
    r"\bleaked\b",
    r"\bbanned\b",
    r"\bdoctors hate\b",
    r"\bone simple trick\b",
    r"\byou will not believe\b",
    r"\bthey do not want you to know\b",
    r"\bcontrol thoughts\b",
    r"\bclick here\b",
    r"\bgovernment bans all\b",
    r"\bglobal weather control\b",
    r"चौंकाने वाला",
    r"चमत्कारी इलाज",
    r"गुप्त दस्तावेज",
    r"डॉक्टर नहीं चाहते",
    r"पूरी तरह ठीक",
    r"रातोंरात",
    r"अफवाह",
    r"यहां क्लिक करें",
    r"विचार नियंत्रित",
]


@dataclass(frozen=True)
class PredictionResult:
    label: str
    prediction: int | None
    model_label: str
    confidence: float
    model_confidence: float
    vocabulary_coverage: float
    word_count: int
    risk_flags: list[str]


def _tokens(text: str) -> list[str]:
    return re.findall(
        r"[^\W\d_](?:[^\W\d_]|'(?=[^\W\d_])){1,}",
        text.lower(),
        flags=re.UNICODE,
    )


def _suspicious_hits(text: str) -> list[str]:
    lowered = text.lower()
    return [
        pattern.replace(r"\b", "").replace("\\", "")
        for pattern in SUSPICIOUS_PATTERNS
        if re.search(pattern, lowered)
    ]


def _word_vocabulary(model) -> set[str]:
    features = model.named_steps["features"]
    for name, transformer in features.transformer_list:
        if name == "word_tfidf":
            return set(transformer.vocabulary_.keys())
    return set()


def vocabulary_coverage(text: str, model) -> float:
    tokens = set(_tokens(text))
    if not tokens:
        return 0.0

    vocabulary = _word_vocabulary(model)
    known_tokens = sum(1 for token in tokens if token in vocabulary)
    return known_tokens / len(tokens)


@lru_cache(maxsize=1)
def load_model():
    with MODEL_PATH.open("rb") as model_file:
        return pickle.load(model_file)


def predict_text(text: str) -> PredictionResult:
    cleaned = text.strip()
    if not cleaned:
        return PredictionResult(
            label="UNCERTAIN",
            prediction=None,
            model_label="UNCERTAIN",
            confidence=0.0,
            model_confidence=0.0,
            vocabulary_coverage=0.0,
            word_count=0,
            risk_flags=["No text was provided."],
        )

    model = load_model()
    probabilities = model.predict_proba([cleaned])[0]
    raw_prediction = int(probabilities.argmax())
    model_confidence = float(probabilities[raw_prediction])
    model_label = LABEL_NAMES[raw_prediction]
    coverage = vocabulary_coverage(cleaned, model)
    word_count = len(_tokens(cleaned))
    suspicious_hits = _suspicious_hits(cleaned)

    risk_flags = []
    final_label = model_label
    final_prediction = raw_prediction
    final_confidence = model_confidence

    if word_count < MIN_WORDS:
        risk_flags.append("Text is too short for a reliable article-level prediction.")
    if coverage < MIN_VOCAB_COVERAGE:
        risk_flags.append("Text is very different from the small training dataset.")
    if model_confidence < MIN_CONFIDENCE:
        risk_flags.append("Model confidence is low.")
    if suspicious_hits:
        risk_flags.append("Sensational or unsupported wording was detected.")

    if model_label == "REAL" and suspicious_hits and model_confidence < 0.88:
        final_label = "FAKE"
        final_prediction = 0
        final_confidence = max(model_confidence, 0.68)
        risk_flags.append("The real-news score was overridden by high-risk wording.")
    elif (
        word_count < MIN_WORDS
        or coverage < MIN_VOCAB_COVERAGE
        or model_confidence < MIN_CONFIDENCE
        or (model_label == "REAL" and model_confidence < MIN_REAL_CONFIDENCE)
    ):
        final_label = "UNCERTAIN"
        final_prediction = None
        final_confidence = 1.0 - abs(0.5 - model_confidence) * 2
        final_confidence = 0.0 if math.isnan(final_confidence) else final_confidence

    return PredictionResult(
        label=final_label,
        prediction=final_prediction,
        model_label=model_label,
        confidence=round(float(final_confidence), 4),
        model_confidence=round(model_confidence, 4),
        vocabulary_coverage=round(coverage, 4),
        word_count=word_count,
        risk_flags=risk_flags,
    )
