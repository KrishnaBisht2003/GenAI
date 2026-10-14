import os
import streamlit as st
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from document_processor import (
    save_uploaded_file, 
    process_and_index_document, 
    delete_document, 
    UPLOAD_DIR, 
    CHROMA_PERSIST_DIR
)

st.title("Document Search & Chat")

# --- SIDEBAR: Document Management ---
st.sidebar.header("Document Management")

# 1. Upload File
uploaded_file = st.sidebar.file_uploader("Upload File", type=["pdf", "docx", "txt"])
if uploaded_file is not None:
    if st.sidebar.button("Upload & Index"):
        with st.spinner("Processing..."):
            file_path = save_uploaded_file(uploaded_file)
            chunks = process_and_index_document(file_path, uploaded_file.name)
            st.sidebar.success(f"Indexed ({chunks} chunks)")

st.sidebar.divider()

# 2. View and Delete Files
st.sidebar.subheader("Uploaded Files")
files = os.listdir(UPLOAD_DIR)

if files:
    for file in files:
        col1, col2 = st.sidebar.columns([3, 1])
        col1.text(file)
        if col2.button("Delete", key=file):
            with st.spinner("Deleting..."):
                delete_document(file)
                st.rerun()
else:
    st.sidebar.text("No documents uploaded.")

# --- MAIN AREA: RAG Chat System ---
st.header("Chat")

# Initialize Vector Store and Local LLM
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=embeddings)
retriever = vector_db.as_retriever(search_kwargs={"k": 3})

llm = Ollama(model="llama3")

# Prompt instructing LLM to rely strictly on retrieved context
prompt_template = ChatPromptTemplate.from_template(
    "Answer the question using primarily the provided context below.\n"
    "If sufficient information is not found in the uploaded documents, "
    "explicitly state: 'Sufficient information was not found in the uploaded documents.'\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}"
)

# Chat History Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display prior messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# User Query Input
user_input = st.chat_input("Ask a question about your documents...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Searching and generating answer..."):
            # 1. Retrieve relevant document chunks
            docs = retriever.invoke(user_input)
            
            # 2. Format context text
            context_text = "\n\n".join(doc.page_content for doc in docs)
            
            # 3. Run prompt -> LLM pipeline
            chain = prompt_template | llm | StrOutputParser()
            answer = chain.invoke({"context": context_text, "question": user_input})
            
            # 4. Extract Sources and Page Numbers
            sources = []
            for doc in docs:
                source_name = doc.metadata.get("source", "Unknown")
                page_num = doc.metadata.get("page", None)
                if page_num is not None:
                    sources.append(f"{source_name} - Page {page_num + 1}")
                else:
                    sources.append(f"{source_name}")
            
            if sources:
                unique_sources = list(set(sources))
                answer += "\n\n**Sources:**\n" + "\n".join([f"- {s}" for s in unique_sources])

            st.write(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})