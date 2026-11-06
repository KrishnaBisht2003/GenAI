import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

from langchain_community.vectorstores import Chroma

import config
from document_processor import (
    load_file_documents,
    chunk_documents,
    compute_file_hash,
    get_embeddings,
    get_document_summary,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("RAG-Ingest")


def load_corpus_manifest() -> dict[str, Any]:
    if config.CORPUS_MANIFEST_PATH.is_file():
        try:
            with open(config.CORPUS_MANIFEST_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not read corpus manifest: {e}")
    return {}


def save_corpus_manifest(manifest: dict[str, Any]) -> None:
    try:
        config.CORPUS_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(config.CORPUS_MANIFEST_PATH, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
    except Exception as e:
        logger.warning(f"Could not write corpus manifest: {e}")


def is_corpus_up_to_date(docs_dir: Path | None = None) -> tuple[bool, list[Path]]:
    """Determine whether server documents match the cached manifest state.

    Assumption: Filesystem metadata (mtime and file size) provides an O(N) stat
    check across directory files. Computing full cryptographic SHA-256 hashes
    on every startup creates severe disk I/O bottlenecks when the corpus grows.
    Potential problem: An administrator could remove the Chroma vector directory
    out-of-band while leaving the manifest file intact; we verify the vector
    directory exists on disk to prevent false cache hits.
    """
    target_dir = docs_dir or config.DOCUMENTS_DIR
    if not target_dir.is_dir():
        return False, []

    supported_extensions = {".pdf", ".docx", ".txt", ".md"}
    disk_files = [
        f for f in target_dir.iterdir()
        if f.is_file() and f.suffix.lower() in supported_extensions and not f.name.startswith(".")
    ]

    manifest = load_corpus_manifest()
    if not manifest or "files" not in manifest:
        return False, disk_files

    if manifest.get("embedding_model") != config.EMBEDDING_MODEL_NAME:
        return False, disk_files

    cached_files = manifest["files"]
    disk_file_names = {f.name for f in disk_files}

    if set(cached_files.keys()) != disk_file_names:
        return False, disk_files

    for f in disk_files:
        cached_info = cached_files.get(f.name)
        if not cached_info:
            return False, disk_files
        stat = f.stat()
        if cached_info.get("size") != stat.st_size or cached_info.get("mtime") != stat.st_mtime:
            return False, disk_files

    if not config.CHROMA_PERSIST_DIR.is_dir():
        return False, disk_files

    return True, disk_files


def get_vector_store() -> Chroma:
    config.CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    embeddings = get_embeddings()
    return Chroma(
        collection_name=config.COLLECTION_NAME,
        persist_directory=str(config.CHROMA_PERSIST_DIR),
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"},
    )


def sync_server_documents(
    docs_dir: Path | None = None,
    force_reindex: bool = False,
) -> dict[str, Any]:
    """Synchronize the vector index with server files using incremental upserts.

    Assumption: Only newly added, modified, or removed files require index mutation.
    Unchanged files reuse existing cached vector embeddings.
    Potential problem: Orphaned chunks from deleted server files will pollute
    retrieval results if not explicitly pruned from the Chroma collection via
    metadata queries.
    """
    target_dir = docs_dir or config.DOCUMENTS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    up_to_date, disk_files = is_corpus_up_to_date(target_dir)

    if up_to_date and not force_reindex:
        manifest = load_corpus_manifest()
        logger.info(
            f"Corpus embeddings are cached and up-to-date ({len(disk_files)} documents, "
            f"{manifest.get('total_chunks', 0)} chunks)."
        )
        return {
            "indexed": 0,
            "skipped": len(disk_files),
            "removed": 0,
            "total_chunks": manifest.get("total_chunks", 0),
            "processed_files": [
                {"name": f.name, "status": "cached"} for f in disk_files
            ],
            "from_cache": True,
        }

    logger.info("Synchronizing vector store with documents on disk...")
    vector_db = get_vector_store()
    collection = vector_db._collection

    indexed_sources: set[str] = set()
    source_to_hash: dict[str, str] = {}

    try:
        existing_data = collection.get(include=["metadatas"])
        if existing_data and existing_data.get("metadatas"):
            for meta in existing_data["metadatas"]:
                if meta and "source" in meta:
                    src = meta["source"]
                    indexed_sources.add(src)
                    if "doc_hash" in meta and src not in source_to_hash:
                        source_to_hash[src] = meta["doc_hash"]
    except Exception as e:
        logger.warning(f"Could not inspect existing index: {e}")

    stats = {
        "indexed": 0,
        "skipped": 0,
        "removed": 0,
        "total_chunks": 0,
        "processed_files": [],
        "from_cache": False,
    }

    manifest_files: dict[str, Any] = {}

    for file_path in disk_files:
        file_name = file_path.name
        file_hash = compute_file_hash(file_path)
        stat = file_path.stat()
        summary = get_document_summary(file_path)

        needs_indexing = (
            force_reindex
            or file_name not in indexed_sources
            or source_to_hash.get(file_name) != file_hash
        )

        if not needs_indexing:
            stats["skipped"] += 1
            stats["processed_files"].append({"name": file_name, "status": "cached"})
            count = len(collection.get(where={"source": file_name})["ids"])
            summary["chunks"] = count
            summary["size"] = stat.st_size
            summary["mtime"] = stat.st_mtime
            manifest_files[file_name] = summary
            continue

        logger.info(f"Indexing document: {file_name}")
        try:
            if file_name in indexed_sources or force_reindex:
                try:
                    collection.delete(where={"source": file_name})
                except Exception:
                    pass

            raw_docs = load_file_documents(file_path)
            chunks = chunk_documents(raw_docs)

            if chunks:
                vector_db.add_documents(chunks)
                stats["indexed"] += 1
                logger.info(f"Indexed {file_name} ({len(chunks)} chunks).")
            else:
                logger.warning(f"No text extracted from {file_name}.")

            stats["processed_files"].append({
                "name": file_name,
                "status": "indexed",
                "chunks": len(chunks),
            })
            summary["chunks"] = len(chunks)
            summary["size"] = stat.st_size
            summary["mtime"] = stat.st_mtime
            manifest_files[file_name] = summary
        except Exception as err:
            logger.error(f"Failed to process {file_name}: {err}", exc_info=True)
            stats["processed_files"].append({
                "name": file_name,
                "status": "error",
                "error": str(err),
            })

    current_disk_names = {f.name for f in disk_files}
    orphaned_sources = indexed_sources - current_disk_names
    for orphaned in orphaned_sources:
        logger.info(f"Purging deleted document from vector index: {orphaned}")
        try:
            collection.delete(where={"source": orphaned})
            stats["removed"] += 1
        except Exception as e:
            logger.warning(f"Error removing orphaned document {orphaned}: {e}")

    total_chunks = collection.count()
    stats["total_chunks"] = total_chunks

    save_corpus_manifest({
        "last_synced": time.time(),
        "embedding_model": config.EMBEDDING_MODEL_NAME,
        "total_chunks": total_chunks,
        "files": manifest_files,
    })

    return stats


def main():
    parser = argparse.ArgumentParser(description="Server RAG Document Ingestion")
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Force re-indexing of all server documents",
    )
    parser.add_argument(
        "--docs-dir",
        type=str,
        default=str(config.DOCUMENTS_DIR),
        help="Path to the server documents directory",
    )
    args = parser.parse_args()

    docs_dir = Path(args.docs_dir).resolve()
    logger.info(f"Checking server documents: {docs_dir}")
    results = sync_server_documents(docs_dir=docs_dir, force_reindex=args.reindex)

    logger.info("=== Ingestion Summary ===")
    logger.info(f"From Cache: {results.get('from_cache', False)}")
    logger.info(f"Indexed: {results['indexed']} documents")
    logger.info(f"Skipped (unchanged): {results['skipped']} documents")
    logger.info(f"Removed: {results['removed']} documents")
    logger.info(f"Total chunks in persistent ChromaDB: {results['total_chunks']}")


if __name__ == "__main__":
    main()
