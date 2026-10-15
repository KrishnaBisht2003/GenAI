import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import streamlit as st


@st.cache_resource(show_spinner=False)
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        encode_kwargs={"batch_size": 64},
    )

def save_uploaded_file(uploaded_file, upload_dir: str | Path) -> str:
    """Save an upload inside the current session's private directory."""
    upload_path = Path(upload_dir)
    upload_path.mkdir(mode=0o700, parents=True, exist_ok=True)
    safe_name = Path(uploaded_file.name.replace("\\", "/")).name
    if not safe_name or safe_name in {".", ".."}:
        raise ValueError("The uploaded file has an invalid name.")

    file_path = upload_path / safe_name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    file_path.chmod(0o600)
    return str(file_path)


def process_and_index_document(
    file_path: str,
    original_file_name: str,
    persist_directory: str | Path,
) -> int:

    """Loads a document, splits it into chunks, and stores it in ChromaDB."""
    
    # 1. Select the appropriate loader based on the file extension
    file_extension = os.path.splitext(original_file_name)[1].lower()
    
    if file_extension == '.pdf':
        loader = PyPDFLoader(file_path)
    elif file_extension == '.docx':
        loader = Docx2txtLoader(file_path)
    elif file_extension == '.txt':
        loader = TextLoader(file_path, encoding='utf-8')
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")
        
    # Extract text and basic metadata (PyPDFLoader auto-captures 'page')
    documents = loader.load()
    
    # Standardize the 'source' metadata to the original file name for clean citations
    for doc in documents:
        doc.metadata['source'] = original_file_name

    # 2. Split the extracted text into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)

    # 3. Initialize the local HuggingFace embedding model
    # "all-MiniLM-L6-v2" is a fast, lightweight model perfect for local RAG
    embeddings = get_embeddings()

    # Keep re-uploads of the same filename from leaving stale chunks behind.
    vector_db = Chroma(
        persist_directory=str(persist_directory),
        embedding_function=embeddings,
    )
    vector_db._collection.delete(where={"source": original_file_name})
    if chunks:
        vector_db.add_documents(chunks)
    
    return len(chunks)


def delete_document(
    filename: str,
    upload_dir: str | Path,
    persist_directory: str | Path,
) -> None:
    """Delete one document and its vectors from the current session."""
    safe_name = Path(filename.replace("\\", "/")).name
    if safe_name != filename:
        raise ValueError("Invalid document name.")

    file_path = Path(upload_dir) / safe_name
    if file_path.is_file():
        file_path.unlink()

    embeddings = get_embeddings()
    vector_db = Chroma(
        persist_directory=str(persist_directory),
        embedding_function=embeddings,
    )
    vector_db._collection.delete(where={"source": filename})