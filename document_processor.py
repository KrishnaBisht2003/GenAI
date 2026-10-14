import os
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Configuration paths
CHROMA_PERSIST_DIR = "chroma_db"
UPLOAD_DIR = "uploaded_documents"

# Ensure the upload directory exists
os.makedirs(UPLOAD_DIR, exist_ok=True)

def save_uploaded_file(uploaded_file):
    """Saves a file uploaded via Streamlit to the local disk."""
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def process_and_index_document(file_path: str, original_file_name: str):

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
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 4. Generate embeddings and store them in a persistent ChromaDB instance
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PERSIST_DIR
    )
    
    return len(chunks)


def delete_document(filename: str):
    """Deletes the file from disk and removes its vectors from ChromaDB."""
    # 1. Delete file from local server folder
    file_path = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(file_path):
        os.remove(file_path)

    # 2. Delete vectors from ChromaDB matching the source filename
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings
    )
    vector_db._collection.delete(where={"source": filename})