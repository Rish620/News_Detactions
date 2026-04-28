# Fake News Detector

AI-powered fake news detection using ML + optional LLM explanations through Ollama.

## Project Structure

```text
Apptad_final_projects/
|-- frontend/          # React frontend
|-- backend/           # FastAPI backend
|-- news_detector/     # Shared ML, data, prediction, and explanation modules
|-- model/             # Trained ML model and metrics
|-- data/              # Dataset
|-- train_model.py     # Model training entrypoint
`-- README.md
```

## Setup Instructions

### 1. Install Backend Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python train_model.py
```

The training script saves:
- `model/model.pkl`
- `model/metrics.json`

Accuracy is measured on a held-out test split and cross-validation from `data/news.csv`.
The current dataset is very small, so the API can return `UNCERTAIN` when pasted web text is too short, too unfamiliar, or weakly supported by the training examples. Add more labeled real-world examples to `data/news.csv` and rerun `python train_model.py` to improve predictions.

### 3. Start Backend Server

```bash
cd backend
python -m uvicorn main:app --reload
```

### 4. Start Ollama (for LLM explanations)

```bash
ollama run mistral
```

### 5. Setup Frontend

```bash
cd frontend
npm install
npm start
```

## API Endpoints

- `GET /` - Health check
- `POST /predict` - Analyze news text
- `GET /metrics` - Show model evaluation metrics

### Example Request

```json
{
  "text": "Scientists discover new planet in habitable zone"
}
```

### Example Response

```json
{
  "label": "UNCERTAIN",
  "prediction": null,
  "confidence": 0.84,
  "model_label": "REAL",
  "model_confidence": 0.71,
  "vocabulary_coverage": 0.06,
  "risk_flags": ["Text is very different from the small training dataset."],
  "explanation": "The model cannot make a reliable fake/real call..."
}
```

## Testing

Test with fake news:
- "SHOCKING: Secret government conspiracy exposed"
- "BREAKING: Fake news article claims false facts"
- "Miracle cure eliminates all diseases overnight"

Test with real news:
- "NASA announces successful space mission to Mars"
- "Stock market closes at record high"

## Prediction Notes

This project classifies writing patterns, not absolute truth. A fake story written in a formal news style can still look real to a small ML model. For better real-world behavior:
- grow `data/news.csv` with hundreds or thousands of labeled examples,
- include full article text, not only short headlines,
- keep recent web examples in both classes,
- treat `UNCERTAIN` as a manual fact-check queue.
