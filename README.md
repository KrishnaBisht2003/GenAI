# Enterprise Knowledge Base RAG Search & Chat System
## Administrator Operations & Deployment Guide

This repository provides an enterprise-grade Retrieval-Augmented Generation (RAG) platform. It features single-folder document management, incremental manifest-based vector caching, dynamic multi-process Ollama model discovery, document structural intelligence, and verified citation tracking.

This guide details server setup, model provisioning, document indexing, and maintenance for system administrators.

---

## 1. System Architecture

```
.
├── documents/                # SINGLE FOLDER: Place all server documents here
├── data/
│   ├── chroma_db/            # Persistent ChromaDB vector store (survives restarts)
│   └── corpus_manifest.json  # Sub-millisecond state cache manifest
├── config.py                 # Central configuration (paths, models, similarity thresholds)
├── document_processor.py     # Multi-format loaders, structural profiler, BGE embeddings
├── ingest.py                 # Server-side ingestion CLI & incremental synchronizer
├── ollama_client.py          # Dynamic endpoint probing, model discovery, token streaming
├── rag_engine.py             # Persistent ChromaDB retriever, query contextualization & citations
├── app.py                    # Streamlit frontend with streaming answers and source cards
├── test_rag_pipeline.py      # Automated end-to-end pipeline verification test suite
├── test_multi_process_ollama.py # Multi-process Ollama discovery & crash recovery test suite
└── requirements.txt          # Python dependencies
```

---

## 2. Prerequisites & Installation

### 2.1 Ollama Service
Ollama handles local LLM inference. Install Ollama on the server from [ollama.com](https://ollama.com).

Start the Ollama daemon:
```bash
ollama serve
```
By default, Ollama listens on `http://localhost:11434`.

### 2.2 Python Virtual Environment
Python 3.10+ is recommended. Set up the virtual environment and install dependencies:

```bash
python3 -m venv searchrag
source searchrag/bin/activate
pip install -r requirements.txt
```

---

## 3. Administrator Model Management

Administrators manage LLMs directly on the server host using the native Ollama CLI. There is no web-based admin PIN or upload portal; the system auto-detects models served by active Ollama instances.

### 3.1 Downloading Models
Download any model supported by Ollama:
```bash
ollama pull llama3:latest
ollama pull mistral:latest
ollama pull qwen2.5:7b
```
As soon as a model download finishes, it is detected by the backend and immediately becomes available in the application's model selection dropdown.

### 3.2 Multi-Process / Multi-Port Serving
The system supports multiple concurrent Ollama daemons running on different ports, GPUs, or network interfaces.

To run a second Ollama instance on another port (e.g., port 11435):
```bash
OLLAMA_HOST=0.0.0.0:11435 ollama serve
```

Configure endpoints in `config.py` or export via the environment:
```bash
export OLLAMA_BASE_URLS="http://localhost:11434,http://localhost:11435"
```

The backend dynamically probes all listed endpoints, aggregates available models, and maps each model to its serving host. When a user sends a query, requests are routed to the specific instance hosting that model.

### 3.3 Dead Process Exclusion & Fault Tolerance
If an Ollama serving process crashes, stops, or terminates:
- Health check probes instantly detect the offline instance (via short 1.5-second timeouts).
- Models exclusively hosted on that dead process are **immediately removed** from user choices.
- If the dead process is restarted, its models automatically reappear in the dropdown on the next poll cycle without restarting the Streamlit application.

---

## 4. Document Management & Vector Ingestion

### 4.1 Document Storage Directory
All server knowledge base documents reside in a single folder:
```
./documents/
```
- **Supported Formats:** `.pdf`, `.docx`, `.txt`, `.md`.
- **Adding Documents:** Copy or move files directly into `./documents/`.
- **Removing Documents:** Delete files directly from `./documents/`.
- **Security:** Users can see document names, page counts, and metadata in the catalog, but cannot download or view raw full documents. Internal server paths and chunk counts are not displayed to clients.

### 4.2 Incremental Manifest Caching
The system avoids expensive reindexing on application startups through an incremental manifest cache stored at `data/corpus_manifest.json`:
- **Unchanged Corpus:** If file names, sizes, and modification timestamps match the manifest, startup completes in **$< 1\text{ ms}$** with zero embedding recomputation.
- **Added Files:** Only new files are chunked and appended to ChromaDB.
- **Modified Files:** Existing vectors for the file are deleted and replaced with fresh embeddings.
- **Removed Files:** Vectors for deleted files are purged from ChromaDB to prevent stale hallucinations.

### 4.3 Manual CLI Ingestion Tool
Administrators can run or force synchronization using [`ingest.py`](file:///Users/unixmaster/ta_duty_iitd/GenAI/ingest.py):

```bash
# Check status and perform incremental synchronization
./searchrag/bin/python ingest.py

# Force full re-indexing of all documents from scratch
./searchrag/bin/python ingest.py --reindex

# Ingest from an alternative documents directory
./searchrag/bin/python ingest.py --docs-dir /path/to/custom/documents
```

---

## 5. Configuration Reference (`config.py`)

All key parameters are centralized in [`config.py`](file:///Users/unixmaster/ta_duty_iitd/GenAI/config.py):

| Parameter | Default Value | Description & Operational Notes |
| :--- | :--- | :--- |
| `DOCUMENTS_DIR` | `./documents` | Directory scanned for knowledge base files. |
| `CHROMA_PERSIST_DIR` | `./data/chroma_db` | Persistent SQLite/HNSW vector storage directory. |
| `CORPUS_MANIFEST_PATH`| `./data/corpus_manifest.json` | JSON manifest tracking file hashes, metadata, and structural outlines. |
| `EMBEDDING_MODEL_NAME`| `BAAI/bge-small-en-v1.5` | Top-tier open-source embedding model (MIT license, 512-token context). |
| `CHUNK_SIZE` | `700` | Target chunk size in characters, preventing token window overflow. |
| `CHUNK_OVERLAP` | `150` | Overlap between adjacent chunks to maintain sentence continuity. |
| `TOP_K_RESULTS` | `5` | Maximum number of candidate chunks retrieved per query. |
| `SIMILARITY_THRESHOLD`| `0.55` | Calibrated cosine similarity cutoff (corresponds to 70%+ normalized match score in the UI). |
| `OLLAMA_BASE_URLS` | `["http://localhost:11434", "http://localhost:11435"]` | List of Ollama daemon endpoints to probe for models. |
| `DEFAULT_NUM_CTX` | `4096` | Context window size allocated in Ollama VRAM per inference request. |

---

## 6. Running the Application

### 6.1 Launch the Streamlit Service
```bash
./searchrag/bin/streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```
- On startup, the app validates the manifest against `./documents/` in $< 1\text{ ms}$.
- Access the web interface at `http://<SERVER_IP>:8501`.

### 6.2 Service Features in UI
- **Model Selector:** Dynamically displays active models across all live Ollama instances.
- **Sync Knowledge Base Button:** Re-evaluates `./documents/` on demand without restarting the server.
- **Verified Sources Shelf:** Displays citations matching the calibrated similarity threshold with excerpt preview drawers.
- **Clean ChatGPT-Style Citations:** Outputs clean in-text markers (`[Source 1]`, `[Source 2]`) linked to verified sources.
- **Document Structure & Overview:** Automatically answers global document questions (e.g., total questions, titles, deadlines) via precomputed structural summaries.

---

## 7. Verification & Testing

Two automated test suites verify system health:

### 7.1 End-to-End Pipeline & Retrieval Test
Verifies Ollama connectivity, document structural intelligence, document-level factual inferences, excerpt retrieval, and out-of-domain rejection:
```bash
./searchrag/bin/python test_rag_pipeline.py
```

### 7.2 Multi-Process Discovery & Crash Recovery Test
Verifies multi-port probing, dead process exclusion, process death simulation, and request routing:
```bash
./searchrag/bin/python test_multi_process_ollama.py
```

---

## 8. Troubleshooting

- **No models appearing in the dropdown:**
  Ensure Ollama is running (`ollama serve` or `curl http://localhost:11434/api/tags`). Download at least one model (`ollama pull llama3:latest`).
- **Embedding model download on air-gapped systems:**
  HuggingFace models are cached in `~/.cache/huggingface/hub`. For offline deployments, download `BAAI/bge-small-en-v1.5` once while connected; the loader uses local cache files when offline.
- **Vector index corruption or schema reset:**
  Remove the Chroma directory and force a reindex:
  ```bash
  rm -rf data/chroma_db data/corpus_manifest.json
  ./searchrag/bin/python ingest.py --reindex
  ```
