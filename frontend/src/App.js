import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

const API_URLS = [
  'http://127.0.0.1:8000/predict',
  'http://localhost:8000/predict'
];

function App() {
  const [text, setText] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const analyzeNews = async () => {
    if (!text.trim()) return;

    setLoading(true);
    try {
      let response;
      let lastError;

      for (const apiUrl of API_URLS) {
        try {
          response = await axios.post(apiUrl, { text });
          break;
        } catch (error) {
          lastError = error;
        }
      }

      if (!response) {
        throw lastError;
      }

      setResult(response.data);
    } catch (error) {
      console.error('Error:', error);
      setResult({
        label: 'ERROR',
        explanation: `Failed to connect to API. Make sure backend is running on port 8000. Details: ${error.message}`
      });
    }
    setLoading(false);
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Fake News Detector</h1>
        <p>AI-powered news verification with ML + LLM</p>
      </header>

      <main className="App-main">
        <div className="input-section">
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Paste news article text here..."
            rows={8}
          />
          <button
            onClick={analyzeNews}
            disabled={loading || !text.trim()}
          >
            {loading ? 'Analyzing...' : 'Analyze News'}
          </button>
        </div>

        {result && (
          <div className="result-section">
            <div className={`result-badge ${result.label === 'REAL' ? 'real' : 'fake'}`}>
              {result.label === 'REAL' ? 'REAL NEWS' : result.label === 'FAKE' ? 'FAKE NEWS' : 'ERROR'}
            </div>
            {typeof result.confidence === 'number' && (
              <div className="confidence">
                Confidence: {(result.confidence * 100).toFixed(1)}%
              </div>
            )}
            <div className="explanation">
              <h3>AI Explanation:</h3>
              <p>{result.explanation}</p>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
