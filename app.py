import streamlit as st
import fitz  # PyMuPDF
import google.generativeai as genai
import chromadb
import docx


# Page configuration
st.set_page_config(page_title="Gemini RAG Assistant", layout="wide", page_icon="✨")


# Initialize session state
def init_session_state():
    defaults = {
        "uploaded_docs": {},
        "chat_history": [],
        "settings_configured": False,
        "api_key": None,
        "selected_model": "gemini-2.0-flash",
        "chroma_client": None,
        "collection": None,
    }
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


init_session_state()

# Configuration
EMBEDDING_MODEL = "models/text-embedding-004"


def extract_text_from_pdf(file):
    """Extract text from PDF"""
    try:
        file.seek(0)
        doc = fitz.open(stream=file.read(), filetype="pdf")
        text = "".join([page.get_text() for page in doc])
        doc.close()
        return text
    except Exception as e:
        st.error(f"Error reading PDF {file.name}: {e}")
        return None


def extract_text_from_docx(file):
    """Extract text from DOCX"""
    try:
        file.seek(0)
        doc = docx.Document(file)
        return "\n".join([paragraph.text for paragraph in doc.paragraphs])
    except Exception as e:
        st.error(f"Error reading DOCX {file.name}: {e}")
        return None


def chunk_text(text, chunk_size=500, overlap=100):
    """Split text into chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        if end < len(text):
            sentence_end = chunk.rfind(".")
            if sentence_end > chunk_size * 0.7:
                end = start + sentence_end + 1
            else:
                last_space = chunk.rfind(" ")
                if last_space > chunk_size * 0.7:
                    end = start + last_space
            chunk = text[start:end]
        chunks.append(chunk.strip())
        next_start = end - overlap
        start = max(next_start, start + 1) if end < len(text) else end
    return [c for c in chunks if c and c.strip()]


def embed_texts(texts):
    """Generate embeddings."""
    if not st.session_state.settings_configured or not st.session_state.api_key:
        st.error("API Key not configured.")
        return None, []

    embeddings = []
    successful_indices = []

    try:
        # Try batch embedding first
        result = genai.embed_content(
            model=EMBEDDING_MODEL, content=texts, task_type="semantic_similarity"
        )
        embeddings = result["embedding"]
        successful_indices = list(range(len(texts)))
    except Exception as e:
        st.warning(f"Batch embedding failed ({e}), trying individually...")
        embeddings = [None] * len(texts)
        successful_indices = []
        for i, text in enumerate(texts):
            try:
                result = genai.embed_content(
                    model=EMBEDDING_MODEL, content=text, task_type="semantic_similarity"
                )
                embeddings[i] = result["embedding"]
                successful_indices.append(i)
            except Exception as inner_e:
                st.error(f"Embedding failed for chunk {i + 1}: {inner_e}")

    return embeddings, successful_indices


def build_prompt(query, contexts):
    """Build prompt for chat."""
    prompt = "You are a helpful AI assistant. Answer the user's question based on the retrieved context.\n\n"

    if contexts:
        prompt += "Context:\n"
        for i, ctx in enumerate(contexts, 1):
            prompt += f"{i}. {ctx}\n\n"
    else:
        prompt += "No relevant context found.\n\n"

    prompt += f"Question: {query}\n\nAnswer:"
    return prompt


def initialize_chromadb():
    """Initialize ChromaDB."""
    try:
        db_path = "./chroma_db"
        client = chromadb.PersistentClient(path=db_path)
        collection = client.get_or_create_collection(
            name="rag_documents", metadata={"hnsw:space": "cosine"}
        )
        return client, collection
    except Exception as e:
        st.error(f"Vector DB Error: {e}")
        return None, None


# Initialize ChromaDB if settings are configured
if st.session_state.settings_configured and st.session_state.chroma_client is None:
    st.session_state.chroma_client, st.session_state.collection = initialize_chromadb()

# Main App
st.title("💬 Gemini RAG Assistant")
st.write("Upload documents and chat with AI-powered insights")

# Sidebar - Configuration
with st.sidebar:
    st.header("⚙️ Configuration")

    # API Key
    api_key_input = st.text_input(
        "Gemini API Key",
        type="password",
        value=st.session_state.api_key or "",
        help="Enter your Google Generative AI API Key.",
    )

    # Model selection
    model_options = [
        "gemini-2.0-flash",
        "gemini-1.5-flash-latest",
        "gemini-1.5-pro-latest",
    ]
    selected_model_input = st.selectbox(
        "Chat Model",
        model_options,
        index=model_options.index(st.session_state.selected_model)
        if st.session_state.selected_model in model_options
        else 0,
    )

    # Save configuration
    if st.button("Save Configuration"):
        if api_key_input:
            try:
                genai.configure(api_key=api_key_input)
                # Test the API key
                genai.get_model(f"models/{selected_model_input}")
                st.session_state.api_key = api_key_input
                st.session_state.selected_model = selected_model_input
                st.session_state.settings_configured = True

                # Initialize ChromaDB
                if st.session_state.chroma_client is None:
                    st.session_state.chroma_client, st.session_state.collection = (
                        initialize_chromadb()
                    )

                if st.session_state.collection is not None:
                    st.success("✅ Configuration saved!")
                    st.rerun()
                else:
                    st.error("Configuration saved, but database initialization failed.")
            except Exception as e:
                st.session_state.settings_configured = False
                st.error(f"API Key/Model Error: {e}")
        else:
            st.warning("⚠️ API Key is required.")

    # Status
    st.divider()
    if st.session_state.settings_configured:
        st.success("🟢 Configured")
    else:
        st.error("🔴 Not Configured")

    st.write(f"**Model:** {st.session_state.selected_model}")
    st.write(
        f"**Database:** {'🟢 Ready' if st.session_state.collection is not None else '🔴 Not Ready'}"
    )

    st.divider()

    # Document Upload
    st.header("📚 Document Management")

    upload_disabled = (
        not st.session_state.settings_configured or st.session_state.collection is None
    )

    uploaded_files = st.file_uploader(
        "Upload Documents",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
        disabled=upload_disabled,
        help="Upload PDF, TXT, or DOCX files",
    )

    # Process uploaded files
    if uploaded_files and not upload_disabled:
        progress_bar = st.progress(0)
        status_text = st.empty()

        for idx, file in enumerate(uploaded_files):
            file_id = f"{file.name}_{file.size}"
            status_text.text(f"Processing {idx + 1}/{len(uploaded_files)}: {file.name}")
            progress_bar.progress((idx + 1) / len(uploaded_files))

            if file_id not in st.session_state.uploaded_docs:
                # Extract text
                text = None
                if file.type == "application/pdf":
                    text = extract_text_from_pdf(file)
                elif (
                    file.type
                    == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                ):
                    text = extract_text_from_docx(file)
                else:  # txt file
                    try:
                        file.seek(0)
                        text = file.read().decode("utf-8")
                    except Exception as e:
                        st.error(f"Error reading {file.name}: {e}")
                        continue

                if text and text.strip():
                    # Chunk and embed
                    chunks = chunk_text(text)
                    if chunks:
                        embeddings, successful_indices = embed_texts(chunks)
                        if embeddings and successful_indices:
                            successful_chunks = [chunks[i] for i in successful_indices]
                            successful_embeddings = [
                                embeddings[i]
                                for i in successful_indices
                                if embeddings[i] is not None
                            ]

                            if successful_chunks and successful_embeddings:
                                # Add to ChromaDB
                                ids = [
                                    f"{file_id}_{i}"
                                    for i in range(len(successful_chunks))
                                ]
                                metadatas = [
                                    {"source": file.name, "file_id": file_id}
                                    for _ in range(len(successful_chunks))
                                ]

                                st.session_state.collection.add(
                                    documents=successful_chunks,
                                    embeddings=successful_embeddings,
                                    ids=ids,
                                    metadatas=metadatas,
                                )

                                st.session_state.uploaded_docs[file_id] = {
                                    "name": file.name,
                                    "chunks": len(successful_chunks),
                                }

        progress_bar.empty()
        status_text.empty()
        if uploaded_files:
            st.success(f"✅ Processed {len(uploaded_files)} file(s)")

    # Display uploaded documents
    if st.session_state.uploaded_docs:
        st.subheader("Uploaded Documents")
        for doc_id, doc_info in st.session_state.uploaded_docs.items():
            col1, col2 = st.columns([4, 1])
            col1.write(f"📄 {doc_info['name']} ({doc_info['chunks']} chunks)")
            if col2.button("🗑️", key=f"del_{doc_id}", help=f"Delete {doc_info['name']}"):
                # Remove from ChromaDB
                try:
                    doc_chunks = st.session_state.collection.get(
                        where={"file_id": doc_id}
                    )
                    if doc_chunks and doc_chunks.get("ids"):
                        st.session_state.collection.delete(ids=doc_chunks["ids"])
                except Exception as e:
                    st.error(f"Error removing from database: {e}")

                # Remove from session state
                del st.session_state.uploaded_docs[doc_id]
                st.rerun()

        st.divider()

        # Document Selection
        st.subheader("Select Documents for Chat")
        available_doc_names = [
            doc_info["name"] for doc_info in st.session_state.uploaded_docs.values()
        ]
        selected_docs = st.multiselect(
            "Choose documents to search:",
            options=available_doc_names,
            default=available_doc_names,
            help="Select which documents to include in your queries",
        )

    st.divider()

    # Clear chat
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

# Main Chat Interface
if not st.session_state.settings_configured:
    st.warning("⚠️ Please configure your API key in the sidebar first.")
elif not st.session_state.uploaded_docs:
    st.info("📄 Upload some documents in the sidebar to get started.")
else:
    # Check if documents are selected
    selected_doc_names = locals().get("selected_docs", [])

    if not selected_doc_names:
        st.warning("📋 Please select at least one document to chat with.")
    else:
        # Display chat history
        for chat in st.session_state.chat_history:
            # User message
            with st.chat_message("user"):
                st.write(chat["query"])

            # Assistant message
            with st.chat_message("assistant"):
                st.write(chat["response"])

                # Show context if available
                if chat.get("contexts"):
                    with st.expander("View Sources"):
                        for i, (context, source) in enumerate(
                            zip(chat["contexts"], chat.get("context_sources", []))
                        ):
                            st.write(f"**Source {i + 1}: {source}**")
                            st.write(
                                context[:300] + "..." if len(context) > 300 else context
                            )
                            st.divider()

        # Chat input
        if prompt := st.chat_input("Ask a question about your documents..."):
            # Add user message to chat history immediately
            # timestamp = datetime.now().strftime("%H:%M:%S")

            # Display user message
            with st.chat_message("user"):
                # st.write(f"**{timestamp}**")
                st.write(prompt)

            # Process query
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        # Generate query embedding
                        query_embeddings, _ = embed_texts([prompt])
                        if query_embeddings and query_embeddings[0] is not None:
                            query_embedding = query_embeddings[0]

                            # Query ChromaDB
                            results = st.session_state.collection.query(
                                query_embeddings=[query_embedding],
                                n_results=5,
                                where={"source": {"$in": selected_doc_names}},
                            )

                            if (
                                results
                                and results.get("documents")
                                and results["documents"][0]
                            ):
                                contexts = results["documents"][0]
                                context_sources = [
                                    meta.get("source", "Unknown")
                                    for meta in results["metadatas"][0]
                                ]

                                # Generate response
                                prompt_text = build_prompt(prompt, contexts)
                                chat_model = genai.GenerativeModel(
                                    st.session_state.selected_model
                                )
                                response = chat_model.generate_content(prompt_text)
                                response_text = response.text

                                # Display the assistant's response
                                st.write(response_text)

                                # Save to chat history
                                st.session_state.chat_history.append(
                                    {
                                        "query": prompt,
                                        "response": response_text,
                                        "contexts": contexts,
                                        "context_sources": context_sources,
                                    }
                                )
                            else:
                                st.write(
                                    "No relevant context found in selected documents."
                                )
                                st.session_state.chat_history.append(
                                    {
                                        "query": prompt,
                                        "response": "Sorry, I couldn't find relevant information in the selected documents.",
                                        "contexts": [],
                                        "context_sources": [],
                                    }
                                )
                        else:
                            st.write("Failed to generate embedding for the query.")
                            st.session_state.chat_history.append(
                                {
                                    "query": prompt,
                                    "response": "Sorry, I couldn't process your query due to an embedding issue.",
                                    "contexts": [],
                                    "context_sources": [],
                                }
                            )
                    except Exception as e:
                        st.error(f"Unexpected error: {e}")
                        st.session_state.chat_history.append(
                            {
                                "query": prompt,
                                "response": f"An error occurred while processing your query: {e}",
                                "contexts": [],
                                "context_sources": [],
                            }
                        )
