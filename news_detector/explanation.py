from .prediction import PredictionResult


def get_llm_explanation(text: str, result: PredictionResult) -> str:
    """Get an optional explanation from a local Ollama model."""
    try:
        import requests

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "mistral",
                "prompt": (
                    "You are explaining a fake-news classifier result. "
                    "Do not claim the story is factually true just because the model says REAL. "
                    f"Final verdict: {result.label}. Model label: {result.model_label}. "
                    f"Model confidence: {result.model_confidence:.1%}. "
                    f"Risk flags: {', '.join(result.risk_flags) or 'none'}. "
                    f"Explain briefly what this means for the text: {text}"
                ),
                "stream": False,
            },
            timeout=30,
        )
        if response.status_code == 200:
            body = response.json()
            if "response" in body:
                return body["response"]
    except Exception:
        pass

    return fallback_explanation(result)


def fallback_explanation(result: PredictionResult) -> str:
    flags = " ".join(result.risk_flags)

    if result.label == "UNCERTAIN":
        return (
            "The model cannot make a reliable fake/real call for this text. "
            f"It predicted {result.model_label} internally with "
            f"{result.model_confidence:.1%} confidence, but the input needs manual fact-checking. "
            f"{flags}"
        ).strip()

    if result.label == "REAL":
        return (
            "The text looks similar to real-news examples in the training data. "
            "This is a style-based prediction, not proof that the claim is true. "
            f"Model confidence: {result.model_confidence:.1%}."
        )

    return (
        "The text contains patterns commonly seen in fake or misleading examples, "
        "such as sensational wording or weakly supported claims. "
        f"Model confidence: {result.model_confidence:.1%}. {flags}"
    ).strip()
