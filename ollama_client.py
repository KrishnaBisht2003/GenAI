"""Ollama Multi-Process Client and Dynamic Model Discovery Engine.

Dynamically discovers and monitors Ollama serving processes.
Models are automatically picked up from active serving processes on the server.
If any Ollama serving process dies or crashes, its models are immediately
excluded from the user choices.
"""

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Generator, Any
import requests

import config

logger = logging.getLogger("RAG-Ollama")


def probe_endpoint(endpoint: str, timeout: float = 1.0) -> dict[str, Any]:
    norm_url = endpoint.rstrip("/")
    try:
        resp = requests.get(f"{norm_url}/api/tags", timeout=timeout)
        if resp.status_code == 200:
            data = resp.json()
            models = [
                m["name"]
                for m in data.get("models", [])
                if isinstance(m, dict) and "name" in m
            ]
            version = "online"
            try:
                v_resp = requests.get(f"{norm_url}/api/version", timeout=0.8)
                if v_resp.status_code == 200:
                    version = v_resp.json().get("version", version)
            except Exception:
                pass

            return {
                "endpoint": norm_url,
                "online": True,
                "version": version,
                "models": sorted(list(set(models))),
                "error": None,
            }
        else:
            return {
                "endpoint": norm_url,
                "online": False,
                "version": None,
                "models": [],
                "error": f"HTTP status {resp.status_code}",
            }
    except Exception as exc:
        return {
            "endpoint": norm_url,
            "online": False,
            "version": None,
            "models": [],
            "error": str(exc),
        }


def get_serving_processes() -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    endpoints = config.OLLAMA_ENDPOINTS

    max_workers = min(len(endpoints), 6) if endpoints else 1
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {
            executor.submit(probe_endpoint, url): url for url in endpoints
        }
        for future in as_completed(future_to_url):
            info = future.result()
            results[info["endpoint"]] = info

    return results


def check_ollama_status() -> dict[str, Any]:
    processes = get_serving_processes()
    online_endpoints = [ep for ep, st in processes.items() if st["online"]]
    dead_endpoints = [ep for ep, st in processes.items() if not st["online"]]

    is_online = len(online_endpoints) > 0
    primary_version = None
    if online_endpoints:
        primary_version = processes[online_endpoints[0]]["version"]

    return {
        "online": is_online,
        "active_processes": len(online_endpoints),
        "total_endpoints": len(processes),
        "online_endpoints": online_endpoints,
        "dead_endpoints": dead_endpoints,
        "version": primary_version,
        "error": None if is_online else "All configured Ollama serving processes are offline.",
        "processes": processes,
    }


def get_available_models_with_endpoints() -> tuple[list[str], dict[str, str]]:
    processes = get_serving_processes()
    model_to_endpoint: dict[str, str] = {}
    all_models: list[str] = []

    for ep, st in processes.items():
        if st["online"] and st["models"]:
            for m in st["models"]:
                if m not in model_to_endpoint:
                    model_to_endpoint[m] = ep
                    all_models.append(m)

    all_models = sorted(list(dict.fromkeys(all_models)))
    return all_models, model_to_endpoint


def get_available_backend_models() -> list[str]:
    models, _ = get_available_models_with_endpoints()
    return models


def get_endpoint_for_model(model: str) -> str:
    models, model_to_endpoint = get_available_models_with_endpoints()
    if model in model_to_endpoint:
        return model_to_endpoint[model]

    base_name = model.split(":")[0] if ":" in model else model
    for m, ep in model_to_endpoint.items():
        if m.startswith(f"{base_name}:") or base_name == m:
            return ep

    return config.OLLAMA_BASE_URL


def stream_chat_response(
    model: str,
    messages: list[dict[str, str]],
    temperature: float | None = None,
) -> Generator[str, None, None]:
    available_models, model_to_endpoint = get_available_models_with_endpoints()

    if not available_models:
        yield "\n\n⚠️ **Ollama Offline:** No active Ollama serving processes are currently reachable."
        return

    target_model = model
    endpoint = None

    if target_model in model_to_endpoint:
        endpoint = model_to_endpoint[target_model]
    else:
        base_name = target_model.split(":")[0] if ":" in target_model else target_model
        for m, ep in model_to_endpoint.items():
            if m.startswith(f"{base_name}:") or base_name == m:
                target_model = m
                endpoint = ep
                break

    if not endpoint:
        fallback_model = available_models[0]
        endpoint = model_to_endpoint[fallback_model]
        logger.warning(
            f"Model '{model}' is not served by any alive process. Falling back to '{fallback_model}' at {endpoint}."
        )
        target_model = fallback_model

    url = f"{endpoint}/api/chat"
    payload = {
        "model": target_model,
        "messages": messages,
        "stream": True,
        "options": {
            "temperature": temperature if temperature is not None else config.LLM_TEMPERATURE,
            "num_ctx": config.OLLAMA_NUM_CTX,
        },
    }

    try:
        response = requests.post(url, json=payload, stream=True, timeout=config.LLM_TIMEOUT)
        response.raise_for_status()

        for line in response.iter_lines():
            if not line:
                continue
            chunk = json.loads(line.decode("utf-8"))
            if "message" in chunk and "content" in chunk["message"]:
                yield chunk["message"]["content"]
            if chunk.get("done", False):
                break
    except requests.exceptions.RequestException as err:
        yield f"\n\n⚠️ **Ollama Service Error:** Failed to communicate with `{target_model}` at `{endpoint}` (process may have died): {err}"


def contextualize_query(
    model: str,
    chat_history: list[dict[str, str]],
    latest_query: str,
) -> str:
    if not chat_history or len(chat_history) < 2:
        return latest_query

    recent_history = chat_history[-4:]
    history_summary = []
    for msg in recent_history:
        role = "User" if msg["role"] == "user" else "Assistant"
        content = msg["content"][:200]
        history_summary.append(f"{role}: {content}")

    prompt = (
        "Given the conversation history and a follow-up question, rewrite the follow-up question "
        "into a single standalone question that contains all necessary context for a document search. "
        "Do not answer the question, just return the standalone question.\n\n"
        f"Conversation History:\n{chr(10).join(history_summary)}\n\n"
        f"Follow-up Question: {latest_query}\n\n"
        "Standalone Question:"
    )

    endpoint = get_endpoint_for_model(model)
    url = f"{endpoint}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_ctx": 2048,
        },
    }

    try:
        resp = requests.post(url, json=payload, timeout=8)
        if resp.status_code == 200:
            result = resp.json().get("response", "").strip()
            result = result.strip('"').strip("'").strip()
            if result and len(result) > 4:
                return result
    except Exception as e:
        logger.debug(f"Query contextualization fallback: {e}")

    return latest_query
