import time
from pathlib import Path
import streamlit as st

import config
import ollama_client
import rag_engine
from ingest import sync_server_documents

st.set_page_config(
    page_title="IIT Delhi RAG Assistant",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1100px;
    }
    
    .app-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.3rem 1.6rem;
        margin-bottom: 1.5rem;
    }
    .app-header h1 {
        font-size: 1.6rem;
        font-weight: 700;
        color: #f8fafc;
        margin: 0;
    }
    .app-header p {
        font-size: 0.92rem;
        color: #94a3b8;
        margin: 4px 0 0 0;
    }
    
    .pill-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 3px 9px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .pill-green {
        background: #064e3b;
        color: #6ee7b7;
        border: 1px solid #059669;
    }
    .pill-blue {
        background: #1e3a8a;
        color: #93c5fd;
        border: 1px solid #3b82f6;
    }
    .pill-purple {
        background: #4c1d95;
        color: #c4b5fd;
        border: 1px solid #7c3aed;
    }
    
    .chatgpt-sources-shelf {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 8px;
        margin-top: 14px;
        margin-bottom: 4px;
        padding-top: 10px;
        border-top: 1px solid #334155;
    }
    .sources-heading {
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        margin-right: 4px;
    }
    .chatgpt-chip {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #1e293b;
        border: 1px solid #475569;
        border-radius: 9999px;
        padding: 4px 11px;
        font-size: 0.8rem;
        color: #cbd5e1;
        transition: all 0.15s ease-in-out;
    }
    .chatgpt-chip:hover {
        border-color: #38bdf8;
        color: #f8fafc;
        background: #28374d;
    }
    .chip-num {
        background: #0284c7;
        color: white;
        border-radius: 50%;
        width: 17px;
        height: 17px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 0.7rem;
        font-weight: 700;
    }
    
    .quote-card {
        background: #0f172a;
        border: 1px solid #334155;
        border-left: 3px solid #38bdf8;
        border-radius: 8px;
        padding: 11px 14px;
        margin-top: 8px;
        margin-bottom: 8px;
    }
    .quote-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.82rem;
        font-weight: 600;
        color: #38bdf8;
        margin-bottom: 6px;
    }
    .quote-body {
        font-size: 0.85rem;
        color: #cbd5e1;
        line-height: 1.5;
        white-space: pre-wrap;
        background: #020617;
        padding: 8px 10px;
        border-radius: 6px;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    }
    
    .catalog-item {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 10px 12px;
        margin-bottom: 8px;
    }
    .catalog-title {
        font-weight: 600;
        font-size: 0.88rem;
        color: #f1f5f9;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .catalog-sub {
        font-size: 0.77rem;
        color: #94a3b8;
        margin-top: 4px;
        display: flex;
        gap: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Dynamically discover healthy Ollama serving processes.
# Assumes models can be hosted across separate instances; if an instance dies, its models are omitted.
ollama_status = ollama_client.check_ollama_status()
available_models = ollama_client.get_available_backend_models()

if available_models:
    if "active_model" not in st.session_state or st.session_state.active_model not in available_models:
        st.session_state.active_model = (
            config.DEFAULT_MODEL if config.DEFAULT_MODEL in available_models else available_models[0]
        )
else:
    st.session_state.active_model = config.DEFAULT_MODEL

# preventing unnecessary Chroma re-indexing overhead on each page reload.
sync_server_documents()

with st.sidebar:
    st.title("Server Knowledge Base")
    st.caption("🔒 Corporate document repository (Secure read-only catalog)")

    kb_overview = rag_engine.get_knowledge_base_overview()
    total_docs = kb_overview["total_documents"]
    st.metric("Documents in Repository", total_docs)

    st.divider()

    st.subheader("Reasoning Model")
    if not ollama_status["online"]:
        st.error("Ollama server offline (no active serving processes reachable)")
    elif not available_models:
        st.warning("No models currently served by active Ollama processes.")
        st.caption("Admins can download models on the server using `ollama pull <model>`")
    else:
        if len(available_models) == 1:
            st.info(f"**Model:** `{available_models[0]}`")
            st.caption("Active reasoning engine")
            active_model = available_models[0]
            st.session_state.active_model = active_model
        else:
            active_model = st.selectbox(
                "Select Model",
                options=available_models,
                index=available_models.index(st.session_state.active_model) if st.session_state.active_model in available_models else 0,
                key="backend_model_picker",
                help="Models dynamically served by running Ollama processes on the server.",
            )
            st.session_state.active_model = active_model

        proc_count = ollama_status.get("active_processes", 1)
        if proc_count > 1:
            st.caption(f"⚡ {proc_count} Ollama serving processes active")

    st.divider()

    st.subheader("Corpus Document Catalog")
    st.caption("Users can view metadata for all indexed documents in the corpus, but raw document text cannot be downloaded.")

    catalog_filter = st.text_input(
        "Filter catalog",
        placeholder="Search document name...",
        label_visibility="collapsed",
    )

    doc_list = kb_overview["documents"]
    if catalog_filter:
        doc_list = [d for d in doc_list if catalog_filter.lower() in d["name"].lower()]

    if doc_list:
        for doc in doc_list:
            icon = "📄" if doc["extension"] == "PDF" else ("📝" if doc["extension"] == "DOCX" else "📑")
            pages_str = f"{doc['pages']} pages" if doc["pages"] != "N/A" else "Standard doc"
            st.markdown(
                f"""
                <div class="catalog-item">
                    <div class="catalog-title">{icon} {doc['name']}</div>
                    <div class="catalog-sub">
                        <span>Format: <strong>{doc['extension']}</strong></span>
                        <span>Size: <strong>{doc['size_kb']} KB</strong></span>
                        <span>Pages: <strong>{pages_str}</strong></span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.info("No documents match your search.")

    st.divider()

    st.subheader("Retrieval Hyperparameters")
    top_k = st.slider(
        "Max Source Excerpts (Top-K)",
        min_value=2,
        max_value=8,
        value=config.DEFAULT_TOP_K,
        step=1,
        help="Higher values allow cross-document synthesis across multiple files.",
    )
    relevance_threshold = st.slider(
        "Source Relevance Threshold (%)",
        min_value=40,
        max_value=85,
        value=config.CITATION_THRESHOLD_PCT,
        step=5,
        help="Only passages with match percentage >= this threshold are cited as sources.",
    )

    st.subheader("Administration & Session")
    if st.button("🔄 Sync Server Documents", use_container_width=True):
        with st.spinner("Checking server repository for new or deleted files..."):
            sync_res = sync_server_documents(force_reindex=False)
            rag_engine.get_knowledge_base_overview(force_refresh=True)
            st.success(
                f"Sync complete: {sync_res['indexed']} newly indexed, "
                f"{sync_res['skipped']} cached, {sync_res['removed']} purged."
            )
            time.sleep(1)
            st.rerun()

    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.markdown(
    f"""
    <div class="app-header">
        <h1>Information Retrieval Assistant</h1>
        <p>Cross-document semantic retrieval, quantitative reasoning, and verified source citations</p>
        <div style="margin-top: 10px;">
            <span class="pill-badge {'pill-green' if ollama_status['online'] else 'pill-blue'}">
                ● Ollama: {'Online' if ollama_status['online'] else 'Offline'}
            </span>
            <span class="pill-badge pill-purple">
                🧠 Model: {st.session_state.active_model}
            </span>
            <span class="pill-badge pill-green">
                📚 {total_docs} Documents
            </span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


def render_chatgpt_sources(citations: list[dict]):
    if not citations:
        return

    chips_html = ['<div class="chatgpt-sources-shelf"><span class="sources-heading">Sources (≥ 70% Match):</span>']
    for cit in citations:
        page_badge = f"p. {cit['page']}" if cit["page"] != "N/A" else "Section"
        chips_html.append(
            f'<div class="chatgpt-chip">'
            f'<span class="chip-num">{cit["index"]}</span>'
            f'<span>{cit["source"]}</span>'
            f'<span style="color: #64748b;">• {page_badge} ({cit["score_pct"]}%)</span>'
            f'</div>'
        )
    chips_html.append('</div>')
    st.markdown("".join(chips_html), unsafe_allow_html=True)

    with st.expander(f"🔍 Cited Portions & Ground Truth Excerpts ({len(citations)} sources with ≥ 70% match)", expanded=False):
        for cit in citations:
            page_text = f"Page {cit['page']}" if cit["page"] != "N/A" else "Document Section"
            st.markdown(
                f"""
                <div class="quote-card">
                    <div class="quote-header">
                        <span>[Source {cit['index']}] 📄 <strong>{cit['source']}</strong> — {page_text}</span>
                        <span style="color: #10b981;">● {cit['score_pct']}% Match</span>
                    </div>
                    <div class="quote-body">{cit['snippet']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "citations" in msg and msg["citations"]:
            render_chatgpt_sources(msg["citations"])

input_disabled = not ollama_status["online"] or not available_models or total_docs == 0
query_to_run = st.chat_input(
    "Ask a question or request cross-document numerical analysis...",
    disabled=input_disabled,
)

if query_to_run:
    st.session_state.messages.append({"role": "user", "content": query_to_run})
    with st.chat_message("user"):
        st.markdown(query_to_run)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving relevant passages and synthesizing figures..."):
            effective_query = query_to_run
            if len(st.session_state.messages) > 1:
                effective_query = ollama_client.contextualize_query(
                    model=st.session_state.active_model,
                    chat_history=st.session_state.messages[:-1],
                    latest_query=query_to_run,
                )

            retrieved_chunks, is_confident = rag_engine.retrieve_relevant_chunks(
                query=effective_query,
                top_k=top_k,
                similarity_threshold_pct=relevance_threshold,
            )

            is_meta = rag_engine.is_structural_or_meta_query(effective_query)

            # Prevent calling the LLM on completely out-of-domain queries when no relevant chunks match,
            # but allow document-level structural queries to pass through to the document overview context.
            if not is_confident and not is_meta:
                fallback_reply = (
                    "I could not find sufficient information in the available documents "
                    f"to answer this question (no relevant passage met the {relevance_threshold}% confidence threshold)."
                )
                st.markdown(fallback_reply)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": fallback_reply,
                    "citations": [],
                })
            else:
                messages = rag_engine.build_chat_messages(
                    query=query_to_run,
                    chunks=retrieved_chunks,
                    chat_history=st.session_state.messages[:-1],
                )

                response_container = st.empty()
                stream = ollama_client.stream_chat_response(
                    model=st.session_state.active_model,
                    messages=messages,
                )
                raw_answer = response_container.write_stream(stream)
                final_answer = rag_engine.clean_response_text(raw_answer)
                if final_answer != raw_answer:
                    response_container.markdown(final_answer)

                # Overview chunks are excluded from source chips so only actual document pages appear as sources.
                specific_sources = [c for c in retrieved_chunks if not c.get("is_overview")]
                citations = rag_engine.format_citations_metadata(specific_sources)

                if citations:
                    render_chatgpt_sources(citations)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": final_answer,
                    "citations": citations,
                })
