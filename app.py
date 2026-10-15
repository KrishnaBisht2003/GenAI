import os
import hashlib
import json
import re
from pathlib import Path
import streamlit as st
from pydantic import BaseModel, ConfigDict, Field
from requests.exceptions import ConnectionError as RequestsConnectionError
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import PydanticOutputParser

from document_processor import (
    save_uploaded_file, 
    process_and_index_document, 
    delete_document, 
    get_embeddings,
)
from ollama_manager import OLLAMA_BASE_URL, list_ollama_models
from session_storage import get_session_workspace

OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3")
OLLAMA_NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", "4096"))
session_id, session_upload_dir, session_chroma_dir = get_session_workspace(
    st.session_state
)

st.title("Document Search & Chat")

# --- SIDEBAR: Document Management ---
st.sidebar.header("Document Management")

# 1. Upload File
uploaded_file = st.sidebar.file_uploader("Upload File", type=["pdf", "docx", "txt"])
if uploaded_file is not None:
    if st.sidebar.button("Upload & Index"):
        with st.spinner("Processing..."):
            file_path = save_uploaded_file(uploaded_file, session_upload_dir)
            source_name = Path(file_path).name
            chunks = process_and_index_document(
                file_path,
                source_name,
                session_chroma_dir,
            )
            st.session_state.rag_history = []
            st.session_state.rag_evidence = {}
            st.sidebar.success(f"Indexed ({chunks} chunks)")

st.sidebar.divider()

# 2. View and Delete Files
st.sidebar.subheader("Uploaded Files")
files = sorted(path.name for path in session_upload_dir.iterdir() if path.is_file())

if files:
    for file in files:
        col1, col2 = st.sidebar.columns([3, 1])
        col1.text(file)
        if col2.button("Delete", key=file):
            with st.spinner("Deleting..."):
                delete_document(file, session_upload_dir, session_chroma_dir)
                st.session_state.rag_history = []
                st.session_state.rag_evidence = {}
                st.rerun()
else:
    st.sidebar.text("No documents uploaded.")

# --- MAIN AREA: RAG Chat System ---
st.header("Chat")

ollama_error = None
try:
    available_models = list_ollama_models()
except RuntimeError as error:
    available_models = []
    ollama_error = str(error)
    st.sidebar.warning(ollama_error)

if available_models:
    if (
        "reasoning_model" in st.session_state
        and st.session_state["reasoning_model"] not in available_models
    ):
        del st.session_state["reasoning_model"]

    default_model_index = (
        available_models.index(OLLAMA_MODEL)
        if OLLAMA_MODEL in available_models
        else 0
    )
    selected_model = st.sidebar.selectbox(
        "Reasoning model",
        options=available_models,
        index=default_model_index,
        key="reasoning_model",
    )
    st.sidebar.caption(f"Currently used for answers: {selected_model}")
else:
    selected_model = OLLAMA_MODEL
    st.sidebar.info("No models are available from the configured Ollama service.")

if available_models:
    st.caption(f"Active reasoning model: `{selected_model}`")
else:
    st.caption(f"Configured model: `{selected_model}` (unavailable)")


class AnswerParagraph(BaseModel):
    text: str = Field(description="One coherent answer paragraph.")
    source_ids: list[int] = Field(
        default_factory=list,
        description="IDs of retrieved sources supporting this paragraph, if identifiable.",
    )


class StructuredAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    clarification_question: str | None = Field(
        default=None,
        description="A concise question to resolve ambiguity before answering.",
    )
    paragraphs: list[AnswerParagraph] = Field(default_factory=list)
    insufficient_information: bool = Field(
        default=False,
        description="Whether the retrieved documents do not fully answer the question.",
    )


def is_clarification_question(text: str) -> bool:
    cleaned_text = text.strip()
    return (
        len(cleaned_text) <= 500
        and "?" in cleaned_text
        and re.match(
            r"^(which|what|who|where|when|why|how|could you|can you|do you|please clarify|please specify)\b",
            cleaned_text,
            flags=re.IGNORECASE,
        ) is not None
    )


def extract_transcript_answer(raw_response: str) -> str | None:
    try:
        payload = json.loads(raw_response)
    except json.JSONDecodeError:
        return None

    if (
        isinstance(payload, dict)
        and isinstance(payload.get("text"), str)
        and isinstance(payload.get("annotations"), list)
    ):
        answer_text = payload["text"].strip()
        return answer_text or None
    return None


def find_supporting_source_ids(answer_text: str, evidence_by_key: dict) -> list[int]:
    normalized_answer = " ".join(answer_text.casefold().split())
    if not normalized_answer:
        return []

    return [
        evidence["source_id"]
        for evidence in evidence_by_key.values()
        if normalized_answer in " ".join(evidence["content"].casefold().split())
    ]

# Initialize Vector Store and Local LLM
embeddings = get_embeddings()
vector_db = Chroma(
    persist_directory=str(session_chroma_dir),
    embedding_function=embeddings,
)
retriever = vector_db.as_retriever(search_kwargs={"k": 6})
indexed_chunk_count = vector_db._collection.count()
st.caption(f"Indexed chunks in this session: {indexed_chunk_count}")
if indexed_chunk_count == 0:
    st.info("Upload and index a document in this session before asking questions.")

llm = Ollama(
    model=selected_model,
    base_url=OLLAMA_BASE_URL,
    format="json",
    num_ctx=OLLAMA_NUM_CTX,
)
output_parser = PydanticOutputParser(pydantic_object=StructuredAnswer)

# Clarifications are part of the schema; source passages enter the conversation only once.
prompt_template = ChatPromptTemplate.from_messages([
    (
        "system",
        "Answer using only the uploaded-document evidence in the conversation. "
        "If the question is ambiguous (for example, several documents contain a story), "
        "ask one concise clarification_question instead of guessing; in that case return "
        "no answer paragraphs. Once clarified, use the conversation's earlier document "
        "evidence. You may summarize and make reasonable inferences supported by that "
        "evidence, but do not add outside knowledge. For broad overview questions, summarize "
        "useful information from the retrieved passages instead of claiming there is not "
        "enough information when relevant facts are present. Cite a paragraph when a source "
        "clearly supports it; source_ids may be empty when a citation cannot be identified. "
        "Do not omit a useful grounded summary just because it is not an exact quotation. "
        "Set insufficient_information to true only when the evidence does not answer the "
        "question.\n\n"
        "{format_instructions}\n\n"
        "Known source IDs:\n{source_catalog}"
    ),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{current_turn}"),
])

# Chat History Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_history" not in st.session_state:
    st.session_state.rag_history = []
if "rag_evidence" not in st.session_state:
    st.session_state.rag_evidence = {}
if "rag_next_source_id" not in st.session_state:
    st.session_state.rag_next_source_id = 1

# Display prior messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# User Query Input
user_input = st.chat_input(
    "Ask a question about your documents...",
    disabled=not available_models or indexed_chunk_count == 0,
)

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Searching and generating answer..."):
            docs = retriever.invoke(user_input)
            st.caption(f"Retrieved {len(docs)} passages for this query.")
            evidence_by_key = st.session_state.rag_evidence
            current_evidence = []
            new_evidence = []
            next_source_id = st.session_state.rag_next_source_id

            for doc in docs:
                source_name = doc.metadata.get("source", "Unknown")
                page_num = doc.metadata.get("page")
                reference = source_name
                if page_num is not None:
                    reference += f", p. {page_num + 1}"
                fingerprint = hashlib.sha256(
                    f"{reference}\0{doc.page_content}".encode("utf-8")
                ).hexdigest()
                evidence = evidence_by_key.get(fingerprint)
                if evidence is None:
                    evidence = {
                        "source_id": next_source_id,
                        "reference": reference,
                        "content": doc.page_content,
                    }
                    evidence_by_key[fingerprint] = evidence
                    new_evidence.append(evidence)
                    next_source_id += 1
                current_evidence.append(evidence)

            st.session_state.rag_next_source_id = next_source_id

            conversation_history = list(st.session_state.rag_history)
            source_references = {
                item["source_id"]: item["reference"]
                for item in evidence_by_key.values()
            }
            source_catalog = "\n".join(
                f"[Source {source_id}: {reference}]"
                for source_id, reference in sorted(source_references.items())
            ) or "No document sources have been retrieved yet."
            format_instructions = output_parser.get_format_instructions()

            def build_current_turn(passages):
                passage_text = "\n\n".join(
                    f"[Source {item['source_id']}: {item['reference']}]\n{item['content']}"
                    for item in passages
                )
                if not passage_text:
                    passage_text = "No new passages were retrieved; use relevant evidence from earlier turns."
                return f"New document passages:\n{passage_text}\n\nUser question: {user_input}"

            current_turn = build_current_turn(new_evidence)
            max_prompt_chars = OLLAMA_NUM_CTX * 3
            def estimate_prompt_chars(history, turn, catalog):
                return (
                    len(format_instructions)
                    + len(catalog)
                    + len(turn)
                    + sum(len(message["content"]) for message in history)
                )

            prompt_chars = estimate_prompt_chars(
                conversation_history,
                current_turn,
                source_catalog,
            )
            context_was_compacted = prompt_chars > max_prompt_chars

            if context_was_compacted:
                conversation_history = conversation_history[-4:]
                evidence_by_key = {
                    hashlib.sha256(
                        f"{item['reference']}\0{item['content']}".encode("utf-8")
                    ).hexdigest(): item
                    for item in current_evidence
                }
                st.session_state.rag_evidence = evidence_by_key
                source_references = {
                    item["source_id"]: item["reference"]
                    for item in current_evidence
                }
                source_catalog = "\n".join(
                    f"[Source {source_id}: {reference}]"
                    for source_id, reference in sorted(source_references.items())
                ) or "No document sources have been retrieved yet."
                current_turn = build_current_turn(current_evidence)
                prompt_chars = estimate_prompt_chars(
                    conversation_history,
                    current_turn,
                    source_catalog,
                )
                while conversation_history and prompt_chars > max_prompt_chars:
                    conversation_history = conversation_history[2:]
                    prompt_chars = estimate_prompt_chars(
                        conversation_history,
                        current_turn,
                        source_catalog,
                    )
                st.session_state.rag_history = conversation_history

            answer = None
            structured_answer = None
            if not docs and not conversation_history:
                answer = "Sufficient information was not found in the uploaded documents."
            else:
                chain = prompt_template | llm
                try:
                    raw_response = chain.invoke({
                        "history": conversation_history,
                        "current_turn": current_turn,
                        "source_catalog": source_catalog,
                        "format_instructions": format_instructions,
                    })
                    # assert False, f'{raw_response}'
                    try:
                        structured_answer = output_parser.parse(raw_response)
                    except OutputParserException:
                        transcript_answer = extract_transcript_answer(raw_response)
                        if transcript_answer:
                            structured_answer = StructuredAnswer(
                                paragraphs=[
                                    AnswerParagraph(
                                        text=transcript_answer,
                                        source_ids=find_supporting_source_ids(
                                            transcript_answer,
                                            evidence_by_key,
                                        ),
                                    )
                                ]
                            )
                        elif is_clarification_question(raw_response):
                            answer = raw_response.strip()
                            structured_answer = None
                        else:
                            answer = (
                                f"Model '{selected_model}' returned output that did not match the answer schema. "
                                "Try again or select a model with stronger JSON and instruction-following support."
                            )
                            structured_answer = None
                except RequestsConnectionError:
                    answer = (
                        "Cannot connect to Ollama at "
                        f"{OLLAMA_BASE_URL}. {ollama_error or 'Check that the Ollama server is running.'}"
                    )

            if structured_answer is not None:
                if structured_answer.clarification_question:
                    answer = structured_answer.clarification_question.strip()
                else:
                    rendered_paragraphs = []
                    for paragraph in structured_answer.paragraphs:
                        text = paragraph.text.strip()
                        references = list(dict.fromkeys(
                            source_references[source_id]
                            for source_id in paragraph.source_ids
                            if source_id in source_references
                        ))
                        if references:
                            text += f" ({'; '.join(references)})"
                        rendered_paragraphs.append(text)

                    if structured_answer.insufficient_information:
                        rendered_paragraphs.append(
                            "Sufficient information was not found in the uploaded documents."
                        )
                    answer = "\n\n".join(rendered_paragraphs) or (
                        "Sufficient information was not found in the uploaded documents."
                    )

                conversation_history.extend([
                    {"role": "user", "content": current_turn},
                    {"role": "assistant", "content": answer},
                ])
                st.session_state.rag_history = conversation_history
            elif answer and is_clarification_question(answer):
                conversation_history.extend([
                    {"role": "user", "content": current_turn},
                    {"role": "assistant", "content": answer},
                ])
                st.session_state.rag_history = conversation_history

            if context_was_compacted:
                st.caption("Earlier turns were shortened to make room for relevant document passages.")
            st.write(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})