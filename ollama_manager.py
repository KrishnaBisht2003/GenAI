import os

import requests


OLLAMA_BASE_URL = os.environ.get(
    "OLLAMA_BASE_URL",
    "http://localhost:11434",
).rstrip("/")


def list_ollama_models() -> list[str]:
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
        response.raise_for_status()
        tags = response.json()
    except (requests.RequestException, ValueError) as error:
        raise RuntimeError(
            f"Cannot reach the Ollama service at {OLLAMA_BASE_URL}."
        ) from error

    if not isinstance(tags, dict) or not isinstance(tags.get("models", []), list):
        raise RuntimeError("The Ollama service returned an invalid model list.")

    return sorted({
        model["name"]
        for model in tags.get("models", [])
        if isinstance(model, dict) and isinstance(model.get("name"), str)
    })