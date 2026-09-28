"""
Streamlit app: ask questions about a PDF, answered only from its content.

Streamlit reruns this whole file on every interaction, so anything expensive
(loading the PDF, embedding chunks) is built once and kept in st.session_state.
"""

import os
import tempfile

import streamlit as st

from rag_pipeline import answer_question, build_chain, build_vectorstore, get_sources

st.set_page_config(page_title="PDF Q&A", page_icon="📄")

# --- Sidebar: API key --------------------------------------------------------

with st.sidebar:
    st.header("Setup")

    api_key = st.text_input(
        "Gemini API key",
        type="password",
        help="Get a free key at aistudio.google.com/apikey",
    )

    st.caption(
        "Your key is used only for this session and is never stored. "
        "The free tier allows about 20 questions per day."
    )

    st.divider()
    st.caption(
        "Answers come only from the uploaded PDF. "
        "If the answer isn't in the document, the app says so instead of guessing."
    )

# --- Main page ---------------------------------------------------------------

st.title("📄 Ask your PDF")
st.write(
    "Upload a PDF and ask questions about it. Answers are drawn only from the "
    "document — anything it doesn't cover is reported as out of scope."
)

uploaded_file = st.file_uploader("Choose a PDF", type="pdf")

if not api_key:
    st.info("Enter your Gemini API key in the sidebar to begin.")
    st.stop()

if uploaded_file is None:
    st.info("Upload a PDF to begin.")
    st.stop()

# --- Build the index once per uploaded file ----------------------------------

# Rebuild only when a different file arrives, not on every rerun.
if st.session_state.get("file_name") != uploaded_file.name:
    status = st.empty()

    # PyPDFLoader needs a real path, so write the upload to a temp file.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        with st.spinner("Reading and indexing the PDF..."):
            vectorstore = build_vectorstore(
                tmp_path,
                api_key,
                progress_callback=status.info,
            )
            st.session_state.vectorstore = vectorstore
            st.session_state.chain = build_chain(vectorstore, api_key)
            st.session_state.file_name = uploaded_file.name
            st.session_state.history = []

        status.empty()
        st.success(f"Indexed **{uploaded_file.name}** — ask away.")

    except Exception as e:
        status.empty()
        st.error(f"Could not index this PDF: {e}")
        st.stop()

    finally:
        os.unlink(tmp_path)

# --- Ask a question ----------------------------------------------------------

question = st.text_input("Your question", placeholder="What is this document about?")

if st.button("Ask", type="primary") and question:
    try:
        with st.spinner("Thinking..."):
            answer = answer_question(st.session_state.chain, question)
            pages = get_sources(st.session_state.vectorstore, question)

        st.session_state.history.insert(0, (question, answer, pages))

    except Exception as e:
        st.error(f"Something went wrong: {e}")

# --- Show the conversation ---------------------------------------------------

for q, a, pages in st.session_state.get("history", []):
    st.divider()
    st.markdown(f"**Q: {q}**")
    st.write(a)
    if pages:
        st.caption("Source pages: " + ", ".join(str(p) for p in pages))
