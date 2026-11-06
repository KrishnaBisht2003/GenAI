import os
import hashlib
import re
from pathlib import Path
from typing import Any
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

import config


@st.cache_resource(show_spinner=False)
def get_embeddings() -> HuggingFaceEmbeddings:
    try:
        return HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL_NAME,
            model_kwargs={"local_files_only": True},
            encode_kwargs={"batch_size": 64, "normalize_embeddings": True},
        )
    except Exception:
        return HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL_NAME,
            encode_kwargs={"batch_size": 64, "normalize_embeddings": True},
        )


def compute_file_hash(file_path: Path | str) -> str:
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return sha256.hexdigest()


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def extract_document_structure(
    documents: list[Document],
    file_name: str,
    total_pages: Any = "N/A",
) -> dict[str, Any]:
    if not documents:
        return {
            "title": file_name,
            "total_questions": 0,
            "questions": [],
            "total_marks": "N/A",
            "due_date": "N/A",
            "overview_text": f"Document: {file_name}",
        }

    full_text = "\n".join([d.page_content for d in documents])
    page1_text = documents[0].page_content

    title = ""
    lines = [l.strip() for l in page1_text.splitlines() if l.strip()]
    if lines:
        cand = lines[0]
        cand = re.sub(r"([a-z0-9A-Z])(Assignment|Homework|Project)", r"\1 - \2", cand)
        if len(lines) > 1:
            second = lines[1]
            if any(k in second.lower() for k in ["assignment", "homework", "part", "report", "overview"]):
                if second.lower() not in cand.lower():
                    cand = f"{cand} - {second}"
        title = re.sub(r"\s+", " ", cand).strip()

    if not title or len(title) < 3:
        title = Path(file_name).stem.replace("_", " ")

    q_patterns = [
        r"(?:^|\n)\s*((?:Question|Problem|Exercise|Task)\s+\d+[^:\n]*:[^\n]+)",
        r"(?:^|\n)\s*((?:Question|Problem|Exercise)\s+\d+[\.\:\-][^\n]+)",
        r"(?:^|\n)\s*(Q\d+[\.\:\-][^\n]+)",
    ]
    detected_questions = []
    seen_q = set()
    for doc in documents:
        p_num = doc.metadata.get("page", 1)
        for p in q_patterns:
            for m in re.finditer(p, doc.page_content, re.IGNORECASE):
                item = m.group(1).strip()
                item_clean = re.sub(r"\s+", " ", item)
                q_num_match = re.search(r"\d+", item_clean)
                q_num = q_num_match.group(0) if q_num_match else item_clean[:10]
                if q_num not in seen_q:
                    seen_q.add(q_num)
                    detected_questions.append({
                        "heading": item_clean,
                        "page": p_num,
                    })

    try:
        detected_questions.sort(key=lambda x: int(re.search(r"\d+", x["heading"]).group(0)))
    except Exception:
        pass

    marks_match = re.search(r"Total\s+Marks\s*:\s*(\d+)", full_text, re.IGNORECASE)
    total_marks = f"{marks_match.group(1)} Marks" if marks_match else "N/A"

    due_match = re.search(r"Due\s*:\s*([^\n\)\.]+)", full_text, re.IGNORECASE)
    due_date = due_match.group(1).strip() if due_match else "N/A"

    pages_display = total_pages if total_pages != "N/A" else len(documents)
    lines_ov = [
        f"Document File: {file_name}",
        f"Title / Subject: {title}",
        f"Total Pages: {pages_display}",
    ]
    if total_marks != "N/A":
        lines_ov.append(f"Total Marks: {total_marks}")
    if due_date != "N/A":
        lines_ov.append(f"Due Date: {due_date}")

    if detected_questions:
        lines_ov.append(f"Questions / Tasks in Document ({len(detected_questions)} questions in total):")
        for idx, q_info in enumerate(detected_questions, start=1):
            lines_ov.append(f"  • {q_info['heading']} (Page {q_info['page']})")
    else:
        sections = re.findall(r"(?:^|\n)\s*((?:Section|Chapter|Part)\s+\d+[^:\n]*:[^\n]+)", full_text, re.IGNORECASE)
        if sections:
            lines_ov.append(f"Sections ({len(sections)} sections in total):")
            for s in sections[:6]:
                lines_ov.append(f"  • {s.strip()}")

    overview_text = "\n".join(lines_ov)

    return {
        "title": title,
        "total_questions": len(detected_questions),
        "questions": detected_questions,
        "total_marks": total_marks,
        "due_date": due_date,
        "overview_text": overview_text,
    }


def load_file_documents(file_path: Path | str) -> list[Document]:
    path = Path(file_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    ext = path.suffix.lower()
    raw_docs: list[Document] = []

    if ext == ".pdf":
        loader = PyPDFLoader(str(path))
        raw_docs = loader.load()
        for doc in raw_docs:
            p = doc.metadata.get("page")
            if p is not None and isinstance(p, int):
                doc.metadata["page"] = p + 1
            else:
                doc.metadata["page"] = 1
    elif ext == ".docx":
        loader = Docx2txtLoader(str(path))
        raw_docs = loader.load()
        for doc in raw_docs:
            doc.metadata["page"] = "N/A"
    elif ext in {".txt", ".md"}:
        loader = TextLoader(str(path), encoding="utf-8")
        raw_docs = loader.load()
        for doc in raw_docs:
            doc.metadata["page"] = "N/A"
    else:
        raise ValueError(f"Unsupported document format: {ext}. Supported: .pdf, .docx, .txt, .md")

    file_name = path.name
    doc_hash = compute_file_hash(path)

    for doc in raw_docs:
        doc.page_content = clean_text(doc.page_content)
        doc.metadata["source"] = file_name
        doc.metadata["doc_hash"] = doc_hash
        doc.metadata["file_type"] = ext.lstrip(".").upper()

    return [d for d in raw_docs if d.page_content.strip()]


def chunk_documents(documents: list[Document]) -> list[Document]:
    if not documents:
        return []

    source = documents[0].metadata.get("source", "doc")
    doc_hash = documents[0].metadata.get("doc_hash", "")
    file_type = documents[0].metadata.get("file_type", "DOC")

    structure = extract_document_structure(documents, source)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks = splitter.split_documents(documents)
    for idx, chunk in enumerate(chunks):
        chunk.page_content = clean_text(chunk.page_content)
        chunk.metadata["chunk_index"] = idx
        page = chunk.metadata.get("page", 1)
        chunk.metadata["chunk_id"] = f"{source}_p{page}_c{idx}"

    overview_content = (
        f"=== DOCUMENT STRUCTURAL OVERVIEW & SUMMARY: {source} ===\n"
        f"{structure['overview_text']}\n\n"
        f"Introductory Excerpt:\n"
        f"{documents[0].page_content[:450]}"
    )

    overview_chunk = Document(
        page_content=clean_text(overview_content),
        metadata={
            "source": source,
            "page": 1,
            "chunk_index": -1,
            "chunk_id": f"{source}_overview",
            "doc_hash": doc_hash,
            "file_type": file_type,
            "is_overview": True,
        },
    )

    return [overview_chunk] + [c for c in chunks if c.page_content.strip()]


def get_document_summary(file_path: Path | str) -> dict[str, Any]:
    path = Path(file_path)
    if not path.is_file():
        return {}

    stat = path.stat()
    ext = path.suffix.lower().lstrip(".").upper()
    size_kb = round(stat.st_size / 1024, 1)

    page_count = "N/A"
    raw_docs = []
    try:
        raw_docs = load_file_documents(path)
        if path.suffix.lower() == ".pdf":
            page_count = len(raw_docs)
    except Exception:
        pass

    structure = extract_document_structure(raw_docs, path.name, page_count)

    return {
        "name": path.name,
        "extension": ext,
        "size_kb": size_kb,
        "pages": page_count,
        "modified_at": stat.st_mtime,
        "hash": compute_file_hash(path),
        "structure": structure,
        "title": structure["title"],
        "total_questions": structure["total_questions"],
    }
