import streamlit as st

from modules.db import init_db, save_message, get_user_history, delete_user_history
from modules.auth_ui import render_login_page
from modules.ocr import extract_text_with_ocr, render_page_image
from modules.vision import describe_image
from modules.pdf_loader import extract_pdf_text
from modules.chunker import split_text
from modules.embeddings import create_embeddings
from modules.vector_store import create_vector_store
from modules.retriever import retrieve_chunks
from modules.llm import generate_answer


st.set_page_config(
    page_title="AI PDF Assistant",
    page_icon="📄",
    layout="wide"
)

# ---------- VISUAL THEME / STYLING (applies to the main app, not
# the login page, which has its own styling in auth_ui.py) ----------

st.markdown(
    """
    <style>

    .stApp {
        background: linear-gradient(180deg, #f7f8fc 0%, #eef1fb 100%);
    }

    .main .block-container {
        animation: fadeIn 0.5s ease-in-out;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(6px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .stChatMessage {
        border-radius: 14px;
        padding: 6px 6px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #eef1fd, #f7f8fc);
        border: 1px solid rgba(99, 102, 241, 0.15);
        border-radius: 12px;
        padding: 12px 8px;
        text-align: center;
        transition: transform 0.15s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
    }

    .source-chip {
        display: inline-block;
        background: linear-gradient(135deg, #e0e7ff, #ede9fe);
        color: #3730a3;
        border-radius: 20px;
        padding: 3px 12px;
        margin: 3px 5px 3px 0;
        font-size: 0.85em;
        font-weight: 500;
    }

    .stButton > button, .stFormSubmitButton > button {
        border-radius: 10px;
        transition: transform 0.12s ease, box-shadow 0.12s ease;
    }
    .stButton > button:hover, .stFormSubmitButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 3px 10px rgba(99, 102, 241, 0.25);
    }

    div[data-testid="stForm"] {
        background-color: rgba(255, 255, 255, 0.7);
        border-radius: 16px;
        padding: 20px 24px;
        border: 1px solid rgba(99, 102, 241, 0.12);
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ---------- DATABASE INIT ----------

init_db()

# ---------- SESSION STATE ----------

if "user" not in st.session_state:
    st.session_state.user = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "document_key" not in st.session_state:
    st.session_state.document_key = None

if "chunks" not in st.session_state:
    st.session_state.chunks = None

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "total_pages_processed" not in st.session_state:
    st.session_state.total_pages_processed = 0

if "confirm_delete_history" not in st.session_state:
    st.session_state.confirm_delete_history = False


# =====================================================================
# LOGIN / SIGNUP GATE
# =====================================================================

if st.session_state.user is None:

    render_login_page()

    st.stop()


# =====================================================================
# MAIN APP (only reached once logged in)
# =====================================================================

# ---------- SIDEBAR ----------

st.sidebar.title("📚 AI PDF Assistant")

st.sidebar.success(f"👤 Logged in as **{st.session_state.user['name']}**")

if st.sidebar.button("🚪 Logout"):
    st.session_state.user = None
    st.session_state.messages = []
    st.session_state.document_key = None
    st.session_state.chunks = None
    st.session_state.vector_store = None
    st.rerun()

st.sidebar.info(
    """
Upload PDFs or images and ask questions about
their contents.

Powered by:
- LangChain
- FAISS
- Sentence Transformers
- Groq Llama (answers)
- Groq Vision (images/diagrams)
- Tesseract OCR (scanned PDFs)
"""
)

with st.sidebar.expander("📜 My saved history", expanded=False):

    history = get_user_history(st.session_state.user["id"], limit=20)

    if not history:
        st.caption("No saved conversations yet.")
    else:
        for item in history:
            role_icon = "👤" if item["role"] == "user" else "🤖"
            st.caption(
                f"{role_icon} {item['content'][:80]}"
                f"{'...' if len(item['content']) > 80 else ''}"
            )

        st.divider()

        if st.button("🗑️ Delete all saved history", key="delete_history_btn"):
            st.session_state.confirm_delete_history = True

        if st.session_state.get("confirm_delete_history"):
            st.warning("This permanently deletes all your saved history.")
            col_yes, col_no = st.columns(2)

            with col_yes:
                if st.button("Yes, delete", key="confirm_delete_yes"):
                    delete_user_history(st.session_state.user["id"])
                    st.session_state.confirm_delete_history = False
                    st.rerun()

            with col_no:
                if st.button("Cancel", key="confirm_delete_no"):
                    st.session_state.confirm_delete_history = False
                    st.rerun()

# ---------- TITLE ----------

st.title("📄 RAG Document Question Answering System")


def render_sources(source_pages, source_chunks, key_prefix):
    """Render a sources list + expandable excerpts for one answer."""

    if not source_pages:
        return

    chips = "".join(
        f"<span class='source-chip'>📄 {source} — Page {page}</span>"
        for source, page in source_pages
    )

    st.markdown(chips, unsafe_allow_html=True)

    if source_chunks:

        with st.expander("View source text", expanded=False):

            for i, chunk in enumerate(source_chunks):

                st.markdown(
                    f"**{chunk['source']} — Page {chunk['page']}**"
                )

                st.write(chunk["text"][:500])

                if i < len(source_chunks) - 1:
                    st.divider()


IMAGE_EXTENSIONS = ("png", "jpg", "jpeg")


def is_image_file(filename):
    return filename.lower().split(".")[-1] in IMAGE_EXTENSIONS


# ---------- FILE UPLOAD ----------

uploaded_files = st.file_uploader(
    "📄 Upload PDF Files or 🖼️ Images",
    type=["pdf", "png", "jpg", "jpeg"],
    accept_multiple_files=True
)

enable_diagram_understanding = st.checkbox(
    "🖼️ Enable diagram/image understanding for PDFs "
    "(slower — describes pages that look mostly visual)",
    value=False
)


if not uploaded_files:

    st.info(
        "👋 Upload one or more PDFs or images above to get started. "
        "Once processed, you can ask questions and get answers "
        "with cited sources and page numbers."
    )

    st.stop()


# Unique key based on filename + size, so two different files that
# happen to share a name (or a re-upload of the same file) are
# still told apart correctly.
current_document_key = tuple(
    (file.name, file.size) for file in uploaded_files
)

# Build disambiguated display labels in case multiple uploaded files
# share the exact same filename.
name_counts = {}
for file in uploaded_files:
    name_counts[file.name] = name_counts.get(file.name, 0) + 1

occurrence_counter = {}
file_labels = {}

for i, file in enumerate(uploaded_files):
    occurrence_counter[file.name] = occurrence_counter.get(file.name, 0) + 1

    if name_counts[file.name] > 1:
        file_labels[i] = f"{file.name} (copy {occurrence_counter[file.name]})"
    else:
        file_labels[i] = file.name


# ---------- PROCESS FILES ONLY WHEN NEW FILES ARE UPLOADED ----------

if st.session_state.document_key != current_document_key:

    st.session_state.messages = []
    st.session_state.document_key = current_document_key

    all_page_texts = []
    failed_files = []
    empty_text_files = []
    ocr_processed_files = []
    image_processed_files = []

    total_files = len(uploaded_files)
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, uploaded_file in enumerate(uploaded_files):

        display_name = file_labels[i]

        # ---------- DIRECT IMAGE UPLOAD ----------
        if is_image_file(uploaded_file.name):

            status_text.text(
                f"🖼️ Analyzing image {display_name} ({i + 1}/{total_files})..."
            )

            try:
                uploaded_file.seek(0)
                image_bytes = uploaded_file.read()

                ext = uploaded_file.name.lower().split(".")[-1]
                mime_type = "image/jpeg" if ext in ("jpg", "jpeg") else "image/png"

                description = describe_image(image_bytes, mime_type=mime_type)

                if description.startswith("⚠️"):
                    failed_files.append((display_name, description))
                else:
                    all_page_texts.append(
                        {
                            "page": 1,
                            "text": description,
                            "source": display_name
                        }
                    )
                    image_processed_files.append(display_name)

            except Exception as e:
                failed_files.append((display_name, str(e)))

            progress_bar.progress((i + 1) / total_files)
            continue

        # ---------- PDF UPLOAD ----------

        status_text.text(
            f"📖 Reading {display_name} ({i + 1}/{total_files})..."
        )

        try:
            text, total_pages, page_texts = extract_pdf_text(
                uploaded_file
            )
        except Exception as e:
            failed_files.append((display_name, str(e)))
            progress_bar.progress((i + 1) / total_files)
            continue

        # Treat as "no usable text" if the list is empty, OR the
        # average amount of text per page is too small to be real
        # content. This catches two cases:
        # 1. A truly empty text layer (all pages blank)
        # 2. A scanned PDF where a scanning app (e.g. CamScanner)
        #    stamped a short watermark as the only "text" on every
        #    page -- technically non-empty, but not real content,
        #    so a simple "is there any text?" check gets fooled.
        total_chars = sum(
            len(page_data["text"].strip()) for page_data in page_texts
        )
        avg_chars_per_page = (
            total_chars / len(page_texts) if page_texts else 0
        )
        has_real_text = avg_chars_per_page >= 100

        if not has_real_text:

            # No meaningful embedded text -- likely a scanned PDF.
            # Fall back to OCR.
            try:
                uploaded_file.seek(0)
                pdf_bytes = uploaded_file.read()

                status_text.text(
                    f"🔍 Running OCR on {display_name} ({i + 1}/{total_files})..."
                )

                page_texts = extract_text_with_ocr(pdf_bytes)

                if page_texts:
                    ocr_processed_files.append(display_name)
                else:
                    empty_text_files.append(display_name)
                    progress_bar.progress((i + 1) / total_files)
                    continue

            except Exception as e:
                failed_files.append((display_name, f"OCR failed: {e}"))
                progress_bar.progress((i + 1) / total_files)
                continue

        # ---------- OPTIONAL: DIAGRAM/IMAGE UNDERSTANDING ----------
        # For pages whose extracted text is very sparse, they're
        # likely mostly a diagram/chart/photo rather than real text.
        # If enabled, describe that page's visual content and append
        # it, so questions like "what does the diagram on page 5
        # show?" can be answered.
        if enable_diagram_understanding:

            try:
                uploaded_file.seek(0)
                pdf_bytes_for_vision = uploaded_file.read()

                for page_data in page_texts:

                    if len(page_data["text"].strip()) < 100:

                        status_text.text(
                            f"🖼️ Describing visuals on {display_name} "
                            f"page {page_data['page']}..."
                        )

                        page_image_bytes = render_page_image(
                            pdf_bytes_for_vision, page_data["page"]
                        )

                        description = describe_image(page_image_bytes)

                        if not description.startswith("⚠️"):
                            page_data["text"] = (
                                page_data["text"]
                                + "\n\n[Image/Diagram Description]: "
                                + description
                            ).strip()

            except Exception:
                # Non-fatal -- if diagram description fails, we still
                # keep whatever text/OCR content we already have.
                pass

        for page_data in page_texts:
            page_data["source"] = display_name

        all_page_texts.extend(page_texts)

        progress_bar.progress((i + 1) / total_files)

    progress_bar.empty()
    status_text.empty()

    for name, error in failed_files:
        st.error(f"❌ Could not process **{name}**: {error}")

    for name in image_processed_files:
        st.info(f"🖼️ **{name}** analyzed using Groq Vision.")

    for name in ocr_processed_files:
        st.info(f"🔍 **{name}** had no usable text layer — processed using OCR instead.")

    for name in empty_text_files:
        st.warning(
            f"⚠️ No extractable text found in **{name}**, "
            "even after attempting OCR. The file may be blank, "
            "corrupted, or too low quality to read."
        )

    if not all_page_texts:

        st.error(
            "None of the uploaded files produced readable content. "
            "Please upload a text-based PDF or a clear image."
        )

        st.session_state.document_key = None
        st.session_state.chunks = None
        st.session_state.vector_store = None
        st.stop()

    with st.spinner("✂️ Creating document chunks..."):
        chunks = split_text(all_page_texts)

    with st.spinner("🧠 Creating embeddings..."):
        embeddings = create_embeddings(chunks)

    with st.spinner("🔎 Building vector store..."):
        vector_store = create_vector_store(embeddings)

    st.session_state.chunks = chunks
    st.session_state.vector_store = vector_store
    st.session_state.total_pages_processed = len(all_page_texts)

if st.session_state.vector_store is None:
    st.stop()

# ---------- DOCUMENT SUMMARY ----------

st.success(f"✅ {len(uploaded_files)} file(s) ready!")

with st.expander("📚 Uploaded files", expanded=False):
    for i, uploaded_file in enumerate(uploaded_files):
        icon = "🖼️" if is_image_file(uploaded_file.name) else "📄"
        st.write(f"{icon} **{file_labels[i]}**")

col1, col2, col3 = st.columns(3)
col1.metric("Files", len(uploaded_files))
col2.metric("Pages processed", st.session_state.total_pages_processed)
col3.metric("Chunks created", len(st.session_state.chunks))

st.divider()

# ---------- QUESTION FORM ----------

with st.form(key="question_form", clear_on_submit=True):

    question = st.text_input(
        "💬 Ask a question about your files",
        placeholder="Example: What does the diagram on page 5 show?"
    )

    submitted = st.form_submit_button("Ask", use_container_width=True)

if st.session_state.messages:
    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.rerun()

# ---------- HANDLE NEW QUESTION ----------

if submitted and question:

    chunks = st.session_state.chunks
    vector_store = st.session_state.vector_store

    question_embedding = create_embeddings([question])[0]

    relevant_chunks = retrieve_chunks(
        question_embedding, vector_store, chunks
    )

    context = "\n\n".join(
        f"Source: {chunk['source']}\nPage: {chunk['page']}\n{chunk['text']}"
        for chunk in relevant_chunks
    )

    chat_history = ""
    for message in st.session_state.messages[-6:]:
        chat_history += f"{message['role']}: {message['content']}\n"

    try:
        with st.spinner("🤖 Searching documents and generating answer..."):
            answer = generate_answer(question, context, chat_history)

    except Exception as e:
        st.error(f"❌ Could not generate an answer: {e}")
        st.stop()

    if "I could not find this information in the uploaded document" in answer:
        source_pages = []
        source_chunks = []
    else:
        source_pages = sorted(
            set((chunk["source"], chunk["page"]) for chunk in relevant_chunks)
        )
        source_chunks = relevant_chunks

    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": source_pages,
            "source_chunks": source_chunks
        }
    )

    document_names_str = ", ".join(file_labels[i] for i in file_labels)

    save_message(
        st.session_state.user["id"], document_names_str, "user", question
    )
    save_message(
        st.session_state.user["id"],
        document_names_str,
        "assistant",
        answer,
        sources=[list(pair) for pair in source_pages]
    )

    st.markdown("### 🤖 Latest Answer")

    with st.chat_message("user", avatar="👤"):
        st.write(question)

    with st.chat_message("assistant", avatar="🤖"):
        st.write(answer)
        render_sources(source_pages, source_chunks, key_prefix="latest")

    st.divider()

# ---------- PREVIOUS CONVERSATION (this session) ----------

if st.session_state.messages:

    st.markdown("### 📜 Previous Conversation")

    if submitted and question:
        previous_messages = st.session_state.messages[:-2]
    else:
        previous_messages = st.session_state.messages

    paired = []
    i = 0
    while i < len(previous_messages) - 1:
        if (
            previous_messages[i]["role"] == "user"
            and previous_messages[i + 1]["role"] == "assistant"
        ):
            paired.append((previous_messages[i], previous_messages[i + 1]))
            i += 2
        else:
            i += 1

    for idx, (user_msg, assistant_msg) in enumerate(reversed(paired)):

        with st.chat_message("user", avatar="👤"):
            st.write(user_msg["content"])

        with st.chat_message("assistant", avatar="🤖"):
            st.write(assistant_msg["content"])
            render_sources(
                assistant_msg.get("sources", []),
                assistant_msg.get("source_chunks", []),
                key_prefix=f"prev_{idx}"
            )