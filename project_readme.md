# Local RAG-Based Document Search Server

An end-to-end local Retrieval-Augmented Generation (RAG) application that lets users upload documents (`.pdf`, `.docx`, `.txt`), index their content into a local vector database, and perform natural language search/chat over the documents using Ollama and Streamlit.

## 📁 Project Structure

```
.
├── uploaded_documents/     # Local directory for raw uploaded files
├── chroma_db/              # Persistent local ChromaDB database storage
├── document_processor.py   # Backend: Text loading, chunking, embedding, & vector deletion
├── app.py                  # Frontend & RAG logic: Streamlit UI, LCEL pipeline, & Ollama integration
├── test_processor.py       # Standalone testing script to verify backend indexing
├── requirements.txt        # Python package dependencies
└── README.md               # Application documentation

```

## 📄 File Overview

### 1. `app.py` (Main Application & Frontend)

* **Role:** Serves as the user interface and overall pipeline orchestrator.

* **Responsibilities:**

  * Renders the Streamlit web layout, sidebar file controls, and interactive chat interface (`st.chat_message`, `st.chat_input`).

  * Maintains chat history across web reloads using Streamlit's `st.session_state`.

  * Integrates ChromaDB retriever with the local Ollama LLM (`llama3`) via LangChain Expression Language (LCEL).

  * Formats final responses by appending clear source citations (filename and 1-indexed page numbers).

### 2. `document_processor.py` (Backend Document Engine)

* **Role:** Handles file I/O, text extraction, chunking, embedding generation, and vector store operations.

* **Key Functions:**

  * `save_uploaded_file()`: Writes Streamlit uploaded file buffers directly to the local `./uploaded_documents/` directory.

  * `process_and_index_document()`: Routes files to appropriate loaders (`PyPDFLoader`, `Docx2txtLoader`, `TextLoader`), splits text using `RecursiveCharacterTextSplitter` (1000 chunk size / 200 overlap), converts text to vectors using HuggingFace `all-MiniLM-L6-v2`, and persists vectors to ChromaDB.

  * `delete_document()`: Synchronizes deletion by removing physical files from disk and executing metadata-based purge requests (`where={"source": filename}`) in ChromaDB.

### 3. `requirements.txt` (Project Dependencies)

* **Role:** Specifies exact Python package requirements needed to run the backend engine, vector store, embeddings, and Streamlit frontend.

### 4. `uploaded_documents/` & `chroma_db/` (Directories)

* **`uploaded_documents/`**: Stores raw source files uploaded by the user.

* **`chroma_db/`**: Automatically generated folder containing persistent SQLite and vector index data managed by ChromaDB.

## ⭐ Key Features

1. **Multi-Format Ingestion:** Supports PDF (`PyPDFLoader`), Word documents (`Docx2txtLoader`), and plain text files (`TextLoader`).

2. **Context-Aware Text Chunking:** Uses `RecursiveCharacterTextSplitter` with natural text boundaries (paragraphs, line breaks, spaces) to prevent cut-off words.

3. **Local Embedding Generation:** Converts chunks into dense vectors using HuggingFace's `all-MiniLM-L6-v2` model.

4. **Persistent Vector Store:** Saves embeddings and metadata in ChromaDB (`chroma_db/`) across application restarts.

5. **Synchronized File Management:** Deleting a document removes both the physical file on disk and its associated vector embeddings from ChromaDB.

6. **Strict RAG Guidance:** Custom prompt engineering forces the local LLM (Ollama / Llama 3) to reply strictly using retrieved context or declare that sufficient information was not found.

7. **Source & Page Citation:** Displays exact source filenames and page numbers (where applicable) alongside each generated answer.

## 🛠️ Prerequisites & Installation

### 1. Install & Configure Ollama

Ollama is required to run local LLMs on your machine.

* **Linux / WSL:**

  ```
  curl -fsSL https://ollama.com/install.sh | sh
  
  ```

* **macOS / Windows:** Download the native installer from [ollama.com/download](https://ollama.com/download).

Pull the Llama 3 model:

```
ollama pull llama3

```

### 2. Python Environment Setup

Create a virtual environment and install dependencies:

```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

```

#### `requirements.txt`

```
streamlit
langchain
langchain-community
langchain-core
langchain-text-splitters
chromadb
sentence-transformers
pypdf
docx2txt

```

## 🚀 Running the Application

Launch the Streamlit interface:

```
streamlit run app.py

```

The application will open automatically in your browser at `http://localhost:8501`.

## 📖 How to Use

1. **Upload Documents:** Use the **Document Management** sidebar to upload PDF, DOCX, or TXT files, then click **Upload & Index**.

2. **View & Delete Documents:** Review uploaded files in the sidebar list. Click **Delete** next to any file to remove it from disk and clear its embeddings from ChromaDB.

3. **Ask Questions:** Type questions into the chat input box. The system retrieves relevant chunks from ChromaDB, feeds them to Ollama, and displays the answer with source references.

## 🔍 Architecture & Data Flow

```
[ Document Upload ]
         │
         ▼
[ Extract Text & Metadata ]  <-- (PyPDF / Docx2txt / TextLoader)
         │
         ▼
[ Chunking (1000 chars, 200 overlap) ]
         │
         ▼
[ HuggingFace Embeddings (all-MiniLM-L6-v2) ]
         │
         ▼
[ Persistent ChromaDB Store ]
         │
         ├──────────────────────────────────────────┐
         │                                          │
   (Search Step)                             (Deletion Step)
         │                                          │
         ▼                                          ▼
[ User Query Vector Match ]              [ Delete disk file & ]
         │                               [ ChromaDB records   ]
         ▼
[ Context Chunks + Question ]
         │
         ▼
[ Ollama (Llama 3) LLM ]
         │
         ▼
[ Answer + Citation Output ]

```