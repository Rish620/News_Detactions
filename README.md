# Fake News Detector

AI-powered fake news detection using ML + LLM (Ollama)

## Project Structure

```
Apptad_final_projects/
├── frontend/          # React frontend
├── backend/           # FastAPI backend
├── model/             # Trained ML model
├── data/              # Dataset
├── train_model.py     # Model training script
└── README.md
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
- `model/vectorizer.pkl`
- `model/metrics.json`

Accuracy is measured on a held-out test split from `data/news.csv`. Add more labeled examples to `data/news.csv` and rerun `python train_model.py` to improve real-world predictions.

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
  "label": "REAL",
  "prediction": 1,
  "confidence": 0.817,
  "explanation": "This appears to be legitimate scientific news..."
}
```

## Testing

Test with fake news:
- "SHOCKING: Secret government conspiracy exposed"
- "BREAKING: Fake news article claims false facts"

Test with real news:
- "NASA announces successful space mission to Mars"
- "Stock market closes at record high"
