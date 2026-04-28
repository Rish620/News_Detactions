import json
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

from .config import LABEL_NAMES, METRICS_PATH, MODEL_DIR, MODEL_PATH
from .data import load_dataset


def build_pipeline() -> Pipeline:
    """Build a text classifier with word and character features."""
    features = FeatureUnion(
        [
            (
                "word_tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                    max_features=8000,
                ),
            ),
            (
                "char_tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    lowercase=True,
                    ngram_range=(3, 5),
                    sublinear_tf=True,
                    max_features=6000,
                ),
            ),
        ]
    )

    return Pipeline(
        [
            ("features", features),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    C=2.0,
                    random_state=42,
                ),
            ),
        ]
    )


def train_and_save() -> dict:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df = load_dataset()
    x = df["text"]
    y = df["label"]

    print("Splitting train/test data...")
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    print("Training model...")
    model = build_pipeline()
    model.fit(x_train, y_train)

    print("Evaluating model...")
    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(
        y_test,
        predictions,
        target_names=[LABEL_NAMES[0], LABEL_NAMES[1]],
        output_dict=True,
        zero_division=0,
    )

    folds = min(5, int(y.value_counts().min()))
    cv_accuracy = None
    if folds >= 2:
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
        cv_accuracy = cross_val_score(build_pipeline(), x, y, cv=cv, scoring="accuracy")

    metrics = {
        "samples": int(len(df)),
        "train_samples": int(len(x_train)),
        "test_samples": int(len(x_test)),
        "test_accuracy": float(accuracy),
        "cross_validation_accuracy_mean": (
            float(cv_accuracy.mean()) if cv_accuracy is not None else None
        ),
        "cross_validation_accuracy_std": (
            float(cv_accuracy.std()) if cv_accuracy is not None else None
        ),
        "classification_report": report,
        "notes": [
            "The dataset is very small, so real-world web articles should be treated cautiously.",
            "The API can return UNCERTAIN when the text is too short, unfamiliar, or weakly supported.",
        ],
    }

    print("Saving model...")
    with MODEL_PATH.open("wb") as model_file:
        pickle.dump(model, model_file)
    with METRICS_PATH.open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)

    return metrics

