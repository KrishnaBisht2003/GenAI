import logging
import json
import re
from pathlib import Path
from typing import Any

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

import config
from document_processor import get_embeddings, get_document_summary

logger = logging.getLogger("RAG-Engine")

_global_vector_store: Chroma | None = None


def get_vector_store() -> Chroma:
    global _global_vector_store
    if _global_vector_store is not None:
        return _global_vector_store

    config.CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    embeddings = get_embeddings()
    _global_vector_store = Chroma(
        collection_name=config.COLLECTION_NAME,
        persist_directory=str(config.CHROMA_PERSIST_DIR),
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"},
    )
    return _global_vector_store


def get_knowledge_base_overview(force_refresh: bool = False) -> dict[str, Any]:
    if not force_refresh and config.CORPUS_MANIFEST_PATH.is_file():
        try:
            with open(config.CORPUS_MANIFEST_PATH, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            cached_files = manifest.get("files", {})
            doc_list = []
            for name, meta in cached_files.items():
                meta_copy = dict(meta)
                meta_copy["name"] = name
                meta_copy["indexed"] = True
                meta_copy["chunk_count"] = meta.get("chunks", 0)
                doc_list.append(meta_copy)
            doc_list.sort(key=lambda x: x["name"].lower())
            return {
                "total_chunks": manifest.get("total_chunks", 0),
                "total_documents": len(doc_list),
                "indexed_documents_count": len(doc_list),
                "documents": doc_list,
                "from_cache": True,
            }
        except Exception as err:
            logger.warning(f"Failed to read cached manifest: {err}. Falling back to live scan.")

    vector_db = get_vector_store()
    collection = vector_db._collection
    total_chunks = collection.count()

    server_files = []
    if config.DOCUMENTS_DIR.is_dir():
        for path in config.DOCUMENTS_DIR.iterdir():
            if path.is_file() and not path.name.startswith("."):
                info = get_document_summary(path)
                info["indexed"] = True
                server_files.append(info)

    server_files.sort(key=lambda x: x["name"].lower())

    return {
        "total_chunks": total_chunks,
        "total_documents": len(server_files),
        "indexed_documents_count": len(server_files),
        "documents": server_files,
        "from_cache": False,
    }


def get_corpus_structural_overview() -> str:
    kb = get_knowledge_base_overview()
    docs = kb.get("documents", [])
    if not docs:
        return "No documents are currently available in the server repository."

    blocks = ["=== SERVER KNOWLEDGE BASE DOCUMENT STRUCTURE & OVERVIEW ==="]
    for idx, doc in enumerate(docs, start=1):
        struct = doc.get("structure", {})
        title = struct.get("title") or doc.get("title") or doc.get("name", "Document")
        pages = doc.get("pages", "N/A")
        total_q = struct.get("total_questions", 0)
        total_marks = struct.get("total_marks", "N/A")
        due_date = struct.get("due_date", "N/A")

        block_lines = [
            f"[Document {idx}: {doc['name']}]",
            f"• Title / Heading: {title}",
            f"• Total Pages: {pages}",
        ]
        if total_marks != "N/A":
            block_lines.append(f"• Total Marks: {total_marks}")
        if due_date != "N/A":
            block_lines.append(f"• Due Date: {due_date}")

        questions = struct.get("questions", [])
        if questions:
            block_lines.append(f"• Questions / Tasks ({len(questions)} questions in total):")
            for q_idx, q in enumerate(questions, start=1):
                block_lines.append(f"  {q_idx}. {q['heading']} (Page {q['page']})")
        elif total_q > 0:
            block_lines.append(f"• Total Questions: {total_q}")

        blocks.append("\n".join(block_lines))

    return "\n\n".join(blocks)


def is_structural_or_meta_query(query: str) -> bool:
    q = query.lower().strip()
    patterns = [
        r"\b(title|name of (this|the) (document|file|assignment|pdf))\b",
        r"\bwhat('?s| is) (this|the) (document|assignment|pdf|file) (about|called|named|titled)\b",
        r"\bhow many questions\b",
        r"\bhow many problems\b",
        r"\bhow many parts\b",
        r"\bhow many pages\b",
        r"\bhow many marks\b",
        r"\btotal marks\b",
        r"\bdue date\b",
        r"\bdeadline\b",
        r"\bsummar(y|ize)\b",
        r"\boverview\b",
        r"\boutline\b",
        r"\blist of questions\b",
        r"\bquestions in (this|the)\b",
    ]
    for p in patterns:
        if re.search(p, q):
            return True
    return False


def retrieve_relevant_chunks(
    query: str,
    top_k: int | None = None,
    similarity_threshold_pct: int | None = None,
) -> tuple[list[dict[str, Any]], bool]:
    k = top_k or config.DEFAULT_TOP_K
    threshold_pct = (
        similarity_threshold_pct
        if similarity_threshold_pct is not None
        else config.CITATION_THRESHOLD_PCT
    )

    vector_db = get_vector_store()
    if vector_db._collection.count() == 0:
        return [], False

    is_meta = is_structural_or_meta_query(query)

    candidate_k = max(k * 2, 14)
    try:
        raw_results = vector_db.similarity_search_with_score(query, k=candidate_k)
    except Exception as e:
        logger.warning(f"Similarity search exception: {e}. Falling back to standard search.")
        raw_docs = vector_db.similarity_search(query, k=k)
        raw_results = [(doc, 0.45) for doc in raw_docs]

    q_lower = query.lower()
    stop_words = {
        "what", "where", "when", "which", "does", "have", "with", "this", "that",
        "from", "about", "there", "some", "tell", "show", "many", "much", "each",
        "their", "your", "were", "been", "being", "here", "them", "into"
    }
    q_tokens = [w for w in re.findall(r"\w+", q_lower) if len(w) > 2 and w not in stop_words]
    subpart_matches = re.findall(r"\([a-zA-Z0-9ivx]+\)", query)

    scored_candidates = []
    seen_texts = set()

    for doc, dist in raw_results:
        text_content = doc.page_content.strip()
        if text_content in seen_texts:
            continue
        seen_texts.add(text_content)

        dense_score = max(0.0, min(1.0, 1.0 - float(dist)))
        text_lower = text_content.lower()

        bonus = 0.0
        if q_tokens:
            kw_hits = sum(1 for tok in q_tokens if tok in text_lower)
            bonus += (kw_hits / len(q_tokens)) * 0.12

        for sp in subpart_matches:
            if sp in text_content:
                bonus += 0.10

        final_score = min(1.0, dense_score + bonus)
        score_pct = int(round(final_score * 100))

        if score_pct < threshold_pct and not is_meta:
            continue

        scored_candidates.append({
            "content": text_content,
            "source": doc.metadata.get("source", "Unknown Document"),
            "page": doc.metadata.get("page", "N/A"),
            "score": final_score,
            "score_pct": score_pct,
            "chunk_id": doc.metadata.get("chunk_id", ""),
            "is_above_threshold": score_pct >= threshold_pct,
            "is_overview": doc.metadata.get("is_overview", False),
        })

    scored_candidates.sort(key=lambda x: x["score"], reverse=True)
    retrieved_chunks = scored_candidates[:k]

    # For structural or overview questions, ensure the opening context / overview chunk is present
    # if not already captured by the candidate ranking.
    if is_meta:
        has_page1 = any(c.get("page") == 1 for c in retrieved_chunks)
        if not has_page1:
            try:
                p1_docs = vector_db.similarity_search(
                    "title assignment question course overview",
                    k=2,
                )
                for p_doc in p1_docs:
                    if p_doc.page_content.strip() not in seen_texts:
                        seen_texts.add(p_doc.page_content.strip())
                        retrieved_chunks.append({
                            "content": p_doc.page_content.strip(),
                            "source": p_doc.metadata.get("source", "Unknown Document"),
                            "page": p_doc.metadata.get("page", 1),
                            "score": 0.65,
                            "score_pct": 65,
                            "chunk_id": p_doc.metadata.get("chunk_id", ""),
                            "is_above_threshold": True,
                            "is_overview": p_doc.metadata.get("is_overview", False),
                        })
            except Exception:
                pass

    retrieved_chunks.sort(key=lambda x: x["score"], reverse=True)
    for idx, c in enumerate(retrieved_chunks, start=1):
        c["source_idx"] = idx

    is_confident = (len(retrieved_chunks) > 0) or is_meta
    return retrieved_chunks, is_confident


def format_context_for_prompt(chunks: list[dict[str, Any]]) -> str:
    if not chunks:
        return "No specific passage excerpts were required for this query."

    context_blocks = []
    for idx, c in enumerate(chunks, start=1):
        c["source_idx"] = idx
        context_blocks.append(f"[Source {idx}]\n{c['content']}")

    return "\n\n".join(context_blocks)


def build_chat_messages(
    query: str,
    chunks: list[dict[str, Any]],
    chat_history: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    structural_overview = get_corpus_structural_overview()
    context_text = format_context_for_prompt(chunks)

    system_content = (
        f"{config.SYSTEM_PROMPT}\n\n"
        f"{structural_overview}\n\n"
        "=== RELEVANT SERVER DOCUMENT EXCERPTS (NUMBERED SOURCES) ===\n"
        f"{context_text}\n\n"
        "INSTRUCTIONS FOR THIS RESPONSE:\n"
        "1. For general factual questions about the document structure (such as the document title, "
        "number of questions or sections, total marks, or document summary), answer directly by inferring "
        "from the Document Structure & Overview above without needing an artificial [Source X] tag.\n"
        "2. For specific detailed questions citing facts or equations from the passages, cite them as [Source 1], [Source 2].\n"
        "3. Do NOT mention filenames or page numbers in your body prose."
    )

    messages = [{"role": "system", "content": system_content}]

    if chat_history:
        for turn in chat_history[-6:]:
            if turn["role"] in {"user", "assistant"}:
                messages.append({"role": turn["role"], "content": turn["content"]})

    messages.append({
        "role": "user",
        "content": f"Question:\n{query}",
    })

    return messages


def clean_response_text(text: str) -> str:
    cleaned = re.sub(r"\[Source\s*(\d+)[^\]]*\]", r"[Source \1]", text)
    cleaned = re.sub(r"(?<!Source\s)\[(\d+)\]", r"[Source \1]", cleaned)
    cleaned = re.split(r"\n(?:\*\*|##\s*)?(?:References|Sources):", cleaned)[0]
    return cleaned.strip()


def format_citations_metadata(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    citations = []
    seen = set()
    for idx, c in enumerate(chunks, start=1):
        key = (c["source"], str(c["page"]), c["content"][:80])
        if key in seen:
            continue
        seen.add(key)
        citations.append({
            "index": idx,
            "label": f"[Source {idx}]",
            "source": c["source"],
            "page": c["page"],
            "score_pct": c["score_pct"],
            "snippet": c["content"],
        })
    return citations
