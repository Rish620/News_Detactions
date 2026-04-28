import React, { useMemo, useState } from 'react';
import axios from 'axios';
import './App.css';

const getApiUrls = () => {
  const host = window.location.hostname || '127.0.0.1';
  const urls = [
    `http://${host}:8001/predict`,
    `http://${host}:8000/predict`,
    'http://127.0.0.1:8001/predict',
    'http://127.0.0.1:8000/predict',
    'http://localhost:8001/predict',
    'http://localhost:8000/predict'
  ];

  return [...new Set(urls)];
};

const EXAMPLES = [
  'Stock market closes at record high',
  'Miracle cure eliminates all diseases overnight',
  'Government announces new tax policy for small businesses',
  'Shocking secret society controls every news channel',
  'भारत सरकार ने नई शिक्षा नीति पर समीक्षा बैठक की',
  'यह घरेलू उपाय तीन दिन में कैंसर को पूरी तरह ठीक कर देता है'
];

function App() {
  const [text, setText] = useState('');
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);

  const stats = useMemo(() => {
    const words = text.trim() ? text.trim().split(/\s+/).length : 0;
    const characters = text.length;
    const sentences = text.trim() ? text.split(/[.!?]+/).filter(Boolean).length : 0;
    return { words, characters, sentences };
  }, [text]);

  const analyzeNews = async () => {
    if (!text.trim() || loading) return;

    setLoading(true);
    setResult(null);

    try {
      let response;
      let lastError;

      for (const apiUrl of getApiUrls()) {
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

      const nextResult = response.data;
      setResult(nextResult);
      setHistory((items) => [
        {
          id: Date.now(),
          text: text.trim(),
          label: nextResult.label,
          confidence: nextResult.confidence,
          modelLabel: nextResult.model_label
        },
        ...items
      ].slice(0, 4));
    } catch (error) {
      console.error('Error:', error);
      setResult({
        label: 'ERROR',
        explanation: `Failed to connect to API. Make sure backend is running on port 8000. Details: ${error.message}`
      });
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
      analyzeNews();
    }
  };

  const confidence = typeof result?.confidence === 'number' ? result.confidence : 0;
  const confidencePercent = Math.round(confidence * 1000) / 10;
  const verdictClass = result?.label ? result.label.toLowerCase() : 'idle';
  const verdictMark = result?.label === 'REAL' ? 'R' : result?.label === 'FAKE' ? 'F' : result?.label === 'UNCERTAIN' ? '?' : '-';
  const characterState = loading ? 'thinking' : verdictClass;
  const characterMessage = loading
    ? 'Scanning signals'
    : result?.label === 'REAL'
      ? 'Looks credible'
      : result?.label === 'FAKE'
        ? 'High risk alert'
        : result?.label === 'UNCERTAIN'
          ? 'Needs review'
          : result?.label === 'ERROR'
            ? 'Connection issue'
            : 'Ready to inspect';
  const webVerification = result?.web_verification;
  const webConfidence = typeof webVerification?.confidence === 'number'
    ? Math.round(webVerification.confidence * 100)
    : 0;

  return (
    <div className="app-shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">ML + LLM News Intelligence</p>
          <h1>Fake News Detector</h1>
        </div>
        <div className="status-pill">
          <span className="status-dot" />
          API Ready
        </div>
      </header>

      <main className="workspace">
        <section className="analyzer-panel">
          <div className="panel-heading">
            <div>
              <p className="eyebrow">Input</p>
              <h2>Analyze a headline or article</h2>
            </div>
            <button className="ghost-button" type="button" onClick={() => setText('')}>
              Clear
            </button>
          </div>

          <textarea
            value={text}
            onChange={(event) => setText(event.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Paste or type news text here..."
            rows={10}
          />

          <div className="stats-row">
            <span>{stats.words} words</span>
            <span>{stats.characters} chars</span>
            <span>{stats.sentences} sentences</span>
          </div>

          <div className="example-grid">
            {EXAMPLES.map((example) => (
              <button
                className="example-chip"
                key={example}
                type="button"
                onClick={() => setText(example)}
              >
                {example}
              </button>
            ))}
          </div>

          <button
            className="primary-button"
            type="button"
            onClick={analyzeNews}
            disabled={loading || !text.trim()}
          >
            {loading ? 'Analyzing...' : 'Analyze News'}
          </button>
        </section>

        <section className={`result-panel ${verdictClass} ${loading ? 'scanning' : ''}`}>
          <div className="panel-heading">
            <div>
              <p className="eyebrow">Verdict</p>
              <h2>{result ? result.label : 'Waiting for text'}</h2>
            </div>
            <div className="verdict-mark">{verdictMark}</div>
          </div>

          <div className={`verdict-character ${characterState}`} aria-label={characterMessage}>
            <div className="character-aura" />
            <div className="character-body">
              <span className="character-antenna" />
              <div className="character-face">
                <span className="character-eye left" />
                <span className="character-eye right" />
                <span className="character-mouth" />
              </div>
              <span className="character-arm left" />
              <span className="character-arm right" />
            </div>
            <div className="character-shadow" />
            <p>{characterMessage}</p>
          </div>

          <div className="confidence-ring" style={{ '--score': `${confidence * 360}deg` }}>
            <div>
              <strong>{result?.label === 'ERROR' ? '!' : `${confidencePercent}%`}</strong>
              <span>{result?.label === 'ERROR' ? 'Check API' : result?.label === 'UNCERTAIN' ? 'uncertainty' : 'confidence'}</span>
            </div>
          </div>

          <div className="explanation-box">
            <h3>Analysis</h3>
            <p>
              {result
                ? result.explanation
                : 'Run a prediction to see the model verdict, confidence score, and generated explanation.'}
            </p>
            {result?.risk_flags?.length > 0 && (
              <div className="risk-list">
                {result.risk_flags.map((flag) => (
                  <span key={flag}>{flag}</span>
                ))}
              </div>
            )}
          </div>

          {webVerification && (
            <div className="web-check-block">
              <div className="web-check-heading">
                <div>
                  <h3>Internet verification</h3>
                  <p>{webVerification.summary}</p>
                </div>
                <span className={`web-status ${webVerification.status.toLowerCase()}`}>
                  {webVerification.status.replace(/_/g, ' ')}
                </span>
              </div>

              <div className="web-meta">
                <span>{webConfidence}% evidence score</span>
                <span>Query: {webVerification.query}</span>
              </div>

              {webVerification.sources?.length > 0 && (
                <div className="source-list">
                  {webVerification.sources.map((source) => (
                    <a
                      href={source.url}
                      key={`${source.url}-${source.title}`}
                      rel="noreferrer"
                      target="_blank"
                    >
                      <strong>{source.title}</strong>
                      <span>
                        {source.domain}
                        {source.published ? ` · ${source.published}` : ''}
                      </span>
                    </a>
                  ))}
                </div>
              )}
            </div>
          )}

          <div className="history-block">
            <h3>Recent checks</h3>
            {history.length === 0 ? (
              <p className="muted">No predictions yet.</p>
            ) : (
              <div className="history-list">
                {history.map((item) => (
                  <button
                    className="history-item"
                    key={item.id}
                    type="button"
                    onClick={() => setText(item.text)}
                  >
                    <span className={`mini-label ${item.label.toLowerCase()}`}>{item.label}</span>
                    <span>{item.text}</span>
                    <strong>{item.label === 'UNCERTAIN' ? item.modelLabel : `${Math.round(item.confidence * 100)}%`}</strong>
                  </button>
                ))}
              </div>
            )}
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;
