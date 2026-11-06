import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Dedicated server repository for all source documents (.pdf, .docx, .txt, .md).
# Assumes this directory is controlled by administrators; the synchronization manifest
# detects file additions, modifications, and deletions by size and mtime without needing full re-scans.
DOCUMENTS_DIR = Path(os.environ.get("RAG_DOCUMENTS_DIR", BASE_DIR / "documents")).resolve()

# Persistent vector index and cache directory.
# If deleted or corrupted, the system falls back to a clean rebuild from DOCUMENTS_DIR on next run.
DATA_DIR = BASE_DIR / "data"
CHROMA_PERSIST_DIR = Path(os.environ.get("RAG_CHROMA_DIR", DATA_DIR / "chroma_db")).resolve()
CORPUS_MANIFEST_PATH = DATA_DIR / "corpus_manifest.json"

# Collection name in ChromaDB.
# Initialized with cosine distance metadata (hnsw:space: cosine); changing distance metrics requires reindexing.
COLLECTION_NAME = os.environ.get("RAG_COLLECTION_NAME", "server_rag_corpus")

# Ollama server endpoints. Supports multiple URLs (comma-separated) for admins running separate
# daemon instances across different ports or GPUs. If an endpoint becomes unreachable,
# its models are excluded from user selection to prevent dropped connections.
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
_raw_endpoints = os.environ.get(
    "OLLAMA_BASE_URLS",
    f"{OLLAMA_BASE_URL},http://localhost:11435",
)
OLLAMA_ENDPOINTS: list[str] = list(
    dict.fromkeys(
        [url.strip().rstrip("/") for url in _raw_endpoints.split(",") if url.strip()]
    )
)

# Preferred default reasoning model.
# If not currently installed or served, the system automatically selects the first available model.
DEFAULT_MODEL = os.environ.get("RAG_DEFAULT_MODEL", "llama3:latest")

# Context window size in tokens allocated for Ollama generation.
# 4096 tokens accommodates multi-document retrieved passages; increasing this raises host GPU VRAM consumption.
OLLAMA_NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", "4096"))

# Generation sampling temperature. Kept low (0.1) to enforce factual grounding and suppress hallucination.
LLM_TEMPERATURE = float(os.environ.get("RAG_TEMPERATURE", "0.1"))

# Request timeout in seconds for Ollama API communication.
# Prevents infinite hangs on large prompts or under heavy compute load.
LLM_TIMEOUT = int(os.environ.get("RAG_LLM_TIMEOUT", "60"))

# Character chunking size and overlap.
# 700 characters fits comfortably within the 512-token context window of BGE-small, avoiding truncation;
# 120 character overlap prevents sentence fragmentation across boundaries.
CHUNK_SIZE = int(os.environ.get("RAG_CHUNK_SIZE", "700"))
CHUNK_OVERLAP = int(os.environ.get("RAG_CHUNK_OVERLAP", "120"))

# Dense embedding model loaded from local cache to avoid external network requests during execution.
EMBEDDING_MODEL_NAME = os.environ.get("RAG_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

# Maximum number of top passage excerpts retrieved per query.
# Balances synthesis across multiple documents against attention dilution in smaller reasoning models.
DEFAULT_TOP_K = int(os.environ.get("RAG_TOP_K", "6"))

# Minimum similarity percentage required to qualify as a cited source.
# Calibrated for BGE cosine similarity space where semantic relevance typically spans 55% to 80%.
CITATION_THRESHOLD_PCT = int(os.environ.get("RAG_CITATION_THRESHOLD_PCT", "55"))
SIMILARITY_THRESHOLD = float(CITATION_THRESHOLD_PCT) / 100.0

SYSTEM_PROMPT = """You are an advanced, analytical knowledge retrieval assistant for enterprise documents.
Your task is to analyze and answer user queries using the provided document structural overview and context excerpts from the secure server.

CRITICAL INSTRUCTIONS:
1. Grounding: Answer using the document structure and context excerpts provided. If the requested information is neither present nor inferable from the documents, respond:
   "I could not find sufficient information in the provided server documents to answer this question."

2. Document-Level & Structural Factual Questions:
   - For factual questions about the document itself (such as title, topic, subject, number of questions or sections, total marks, due date, or document summary), answer factually by inferring directly from the Document Structure & Overview and context.
   - For these general document-level factual inferences, state the answer clearly without needing artificial [Source X] tags.

3. Detailed Questions & Citation Rule:
   - For specific factual questions supported by individual text passages, refer to sources in your answer using [Source 1], [Source 2], etc. (e.g. "...allocated $2,200,000 [Source 1]...").
   - Do NOT mention document filenames, page numbers, or match relevance percentages in your body prose (they are automatically displayed in the sources section at the bottom).

4. Cross-Document Synthesis & Quantitative Analysis:
   - When a query pertains to information across multiple documents (e.g. budgets, numbers, contractor costs, material allocations, phases), analyze and cross-reference data from ALL relevant retrieved passages.
   - Clarify how different numbers relate, distinguishing direct labor costs from material allocations, breakdown sub-items, or calculating totals.
   - Present quantitative breakdowns with clean markdown tables or bullet points.
"""
