from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import pickle

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3003",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3003",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import os

# Load model and vectorizer (from parent directory)
model_dir = os.path.join(os.path.dirname(__file__), "..", "model")
model = pickle.load(open(os.path.join(model_dir, "model.pkl"), "rb"))
vectorizer = pickle.load(open(os.path.join(model_dir, "vectorizer.pkl"), "rb"))
metrics_path = os.path.join(model_dir, "metrics.json")

class NewsRequest(BaseModel):
    text: str

def get_llm_explanation(text: str, prediction: int, confidence: float) -> str:
    """Get explanation from Ollama LLM"""
    try:
        import requests
        label = "real" if prediction == 1 else "fake"
        response = requests.post("http://localhost:11434/api/generate", json={
            "model": "mistral",
            "prompt": (
                f"The ML model predicts this news is {label} with "
                f"{confidence:.1%} confidence. Explain briefly why the text "
                f"may look {label}: {text}"
            ),
            "stream": False
        }, timeout=30)
        if response.status_code == 200:
            result = response.json()
            if "response" in result:
                return result["response"]
    except Exception as e:
        pass
    
    # Provide a fallback explanation based on ML prediction
    if prediction == 1:
        return f"This news appears to be REAL based on the ML model ({confidence:.1%} confidence). The text uses factual, institutional, or report-style language commonly found in legitimate news."
    else:
        return f"This news appears to be FAKE based on the ML model ({confidence:.1%} confidence). The text includes sensational, unsupported, or hoax-like wording commonly found in misinformation."

@app.post("/predict")
def predict(data: NewsRequest):
    text = data.text
    
    # ML Prediction
    X = vectorizer.transform([text])
    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]
    confidence = float(probabilities[int(prediction)])
    
    # LLM Explanation
    explanation = get_llm_explanation(text, int(prediction), confidence)
    
    return {
        "label": "FAKE" if prediction == 0 else "REAL",
        "prediction": int(prediction),
        "confidence": round(confidence, 4),
        "explanation": explanation
    }

@app.get("/")
def root():
    return {"message": "Fake News Detection API is running!"}

@app.get("/metrics")
def metrics():
    if not os.path.exists(metrics_path):
        return {"message": "Train the model to generate metrics.json"}

    with open(metrics_path, "r", encoding="utf-8") as metrics_file:
        return json.load(metrics_file)
