
from news_detector.training import train_and_save
=======
import json
import os
import pickle

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


DATA_PATH = os.path.join("data", "news.csv")
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")

# Load dataset and clean missing, empty, and duplicate values

def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""]
    df["label"] = df["label"].astype(int)
    return df.drop_duplicates(subset=["text"])


  #Main function for model trainin and evaluation
def main() -> None:

    metrics = train_and_save()

    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Step 1: Loading and cleaning dataset...")
    df = load_dataset()
    X = df["text"]
    y = df["label"]

    print("Step 2: Splitting train and test data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    print("Step 3: Applying TF-IDF vectorization...")
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
        max_features=5000,
    )
    X_train_vectors = vectorizer.fit_transform(X_train)
    X_test_vectors = vectorizer.transform(X_test)

    print("Step 4: Training Logistic Regression model...")
    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        C=8.0,
        random_state=42,
    )
    model.fit(X_train_vectors, y_train)

    print("Step 5: Evaluating model performance...")
    predictions = model.predict(X_test_vectors)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(
        y_test,
        predictions,
        target_names=["FAKE", "REAL"],
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "samples": int(len(df)),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "test_accuracy": accuracy,
        "classification_report": report,
    }

    print("Step 6: Saving model, vectorizer, and metrics...")
    with open(MODEL_PATH, "wb") as model_file:
        pickle.dump(model, model_file)
    with open(VECTORIZER_PATH, "wb") as vectorizer_file:
        pickle.dump(vectorizer, vectorizer_file)
    with open(METRICS_PATH, "w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)


    print("Model training complete!")
    print(f"Total samples: {metrics['samples']}")
    print(f"Train samples: {metrics['train_samples']}")
    print(f"Test samples: {metrics['test_samples']}")
    print(f"Test accuracy: {metrics['test_accuracy']:.2%}")
    if metrics["cross_validation_accuracy_mean"] is not None:
        print(
            "Cross-validation accuracy: "
            f"{metrics['cross_validation_accuracy_mean']:.2%} "
            f"+/- {metrics['cross_validation_accuracy_std']:.2%}"
        )


if __name__ == "__main__":
    main()
