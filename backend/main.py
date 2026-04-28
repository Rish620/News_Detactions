import json
import os
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from news_detector.config import METRICS_PATH
from news_detector.explanation import get_llm_explanation
from news_detector.prediction import predict_text
from news_detector.verification import verify_with_web

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3003",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3003",
    ],
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1|\[::1\]|192\.168\.\d+\.\d+):\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class NewsRequest(BaseModel):
    text: str


@app.post("/predict")
def predict(data: NewsRequest):
    result = predict_text(data.text)
    verification = verify_with_web(data.text)
    explanation = get_llm_explanation(data.text, result)

    return {
        "label": result.label,
        "prediction": result.prediction,
        "confidence": result.confidence,
        "model_label": result.model_label,
        "model_confidence": result.model_confidence,
        "vocabulary_coverage": result.vocabulary_coverage,
        "word_count": result.word_count,
        "risk_flags": result.risk_flags,
        "web_verification": verification.to_dict(),
        "explanation": explanation,
    }

@app.get("/")
def root():
    return {"message": "Fake News Detection API is running!"}

@app.get("/metrics")
def metrics():
    if not METRICS_PATH.exists():
        return {"message": "Train the model to generate metrics.json"}

    with METRICS_PATH.open("r", encoding="utf-8") as metrics_file:
        return json.load(metrics_file)
