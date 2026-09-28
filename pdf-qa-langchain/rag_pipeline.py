"""
RAG pipeline for PDF question-answering.

Loads a PDF, splits it into chunks, embeds them into a FAISS index, and builds
a chain that answers questions using ONLY the PDF's content.

All logic lives here so the Streamlit app (app.py) only handles the UI.
"""

import time

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Configuration -----------------------------------------------------------

CHUNK_SIZE = 1400
CHUNK_OVERLAP = 150
EMBEDDING_MODEL = "models/gemini-embedding-001"
CHAT_MODEL = "gemini-3.5-flash-lite"
SCORE_THRESHOLD = 0.5
TOP_K = 3

# Gemini free tier allows 100 embedding requests per minute.
# Stay under it by embedding in batches with a pause between them.
EMBED_BATCH_SIZE = 90
EMBED_PAUSE_SECONDS = 60

PROMPT_TEMPLATE = """You answer questions about a specific document, using only the text provided to you below.

Rules:
- Use only the information in the context below to answer the question.
- Do not use any outside knowledge, even if you know the answer.
- If the context does not contain the answer, reply exactly: "Sorry, this information is not available in given pdf"
- Do not guess, infer beyond the context, or fill in gaps from memory.

Context:
{context}

Question:
{question}

Answer:"""


# --- Pipeline steps ----------------------------------------------------------

def load_and_split(pdf_path):
    """Read the PDF and split it into overlapping chunks.

    Returns a list of Document objects, each carrying its page number
    in .metadata so answers can cite their source.
    """
    pages = PyPDFLoader(pdf_path).load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(pages)


def build_vectorstore(pdf_path, api_key, progress_callback=None):
    """Load the PDF, embed its chunks, and return a FAISS vector store.

    progress_callback: optional function(message) used by the app to show
    status while a long embedding job runs.
    """
    chunks = load_and_split(pdf_path)

    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=api_key,
    )

    def report(message):
        if progress_callback:
            progress_callback(message)

    # First batch creates the index.
    first = chunks[:EMBED_BATCH_SIZE]
    report(f"Embedding chunks 1-{len(first)} of {len(chunks)}...")
    vectorstore = FAISS.from_documents(first, embeddings)

    # Remaining batches are added, pausing to respect the rate limit.
    for start in range(EMBED_BATCH_SIZE, len(chunks), EMBED_BATCH_SIZE):
        end = min(start + EMBED_BATCH_SIZE, len(chunks))
        report(f"Waiting {EMBED_PAUSE_SECONDS}s for the rate limit to reset...")
        time.sleep(EMBED_PAUSE_SECONDS)

        report(f"Embedding chunks {start + 1}-{end} of {len(chunks)}...")
        vectorstore.add_documents(chunks[start:end])

    report(f"Done — {len(chunks)} chunks indexed.")
    return vectorstore


def format_docs(docs):
    """Join retrieved chunks into the single string the prompt expects."""
    return "\n\n".join(d.page_content for d in docs)


def build_chain(vectorstore, api_key):
    """Wire retriever, prompt, model and parser into a RAG chain.

    The score-threshold retriever returns NO documents when nothing is
    relevant, so out-of-scope questions arrive with empty context and the
    refusal is forced by the data rather than left to the model's judgment.
    """
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": TOP_K, "score_threshold": SCORE_THRESHOLD},
    )

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    llm = ChatGoogleGenerativeAI(
        model=CHAT_MODEL,
        google_api_key=api_key,
    )

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


def answer_question(chain, question):
    """Run one question through the chain and return the answer text."""
    return chain.invoke(question)


def get_sources(vectorstore, question):
    """Return the page numbers the answer was drawn from, for citations."""
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": TOP_K, "score_threshold": SCORE_THRESHOLD},
    )
    docs = retriever.invoke(question)
    return sorted({d.metadata.get("page_label", "?") for d in docs})
