import os
import tempfile
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.callbacks import get_openai_callback
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI


def load_api_key():
    """Load and verify the OpenAI API key from environment."""
    load_dotenv()
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        st.error(
            "🚨 Missing environment variable: OPENAI_API_KEY."
            " Please create a .env file or set it in your environment."
        )
        st.stop()
    return key

@st.cache_data
def extract_and_split(files, chunk_size=1000, chunk_overlap=200):
    """Extract text from PDFs, split into chunks, return list of docs."""
    docs = []
    for uploaded in files:
        # Save to temp file for PdfReader
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded.read())
            path = tmp.name
        reader = PdfReader(path)
        text = "".join([page.extract_text() or "" for page in reader.pages])
        if not text.strip():
            continue
        splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        for chunk in splitter.split_text(text):
            docs.append({
                "text": chunk,
                "metadata": {"source": uploaded.name}
            })
    return docs


@st.cache_resource
def get_vector_store(docs):
    """Embed chunks and store in a FAISS vector store."""
    texts = [d["text"] for d in docs]
    metadatas = [d["metadata"] for d in docs]
    embeddings = OpenAIEmbeddings()
    return FAISS.from_texts(texts, embeddings, metadatas=metadatas)


class StreamHandler(BaseCallbackHandler):
    """Stream new LLM tokens into a Streamlit container."""
    def __init__(self, container):
        self.container = container
        self.response = ""

    def on_llm_new_token(self, token: str, **kwargs):
        self.response += token
        self.container.markdown(self.response)


def stream_answer(question: str, vector_store):
    """Perform similarity search and stream LLM answer."""
    container = st.empty()
    handler = StreamHandler(container)
    manager = CallbackManager([handler])
    llm = OpenAI(
        model="gpt-4o-mini", streaming=True, callback_manager=manager
    )
    chain = load_qa_chain(llm, chain_type="stuff")
    docs = vector_store.similarity_search(question, k=3)
    with get_openai_callback() as cb:
        answer = chain({"input_documents": docs, "question": question})
    return answer, docs


def main():
    # Validate API key
    load_api_key()

    # UI setup
    st.set_page_config(page_title="PDF QA Demo", layout="wide")
    st.title("📄💬 PDF Q&A Demo")
    st.write(
        "Upload PDF file(s) on the left, then ask any question below.",
        "Answers stream back in real time, with citations provided."
    )

    # Sidebar settings
    debug = st.sidebar.checkbox("Show debug info")

    # File upload
    uploaded_files = st.file_uploader(
        "Upload PDF(s)", type="pdf", accept_multiple_files=True
    )

    # Process upload
    if uploaded_files:
        with st.spinner("Extracting and splitting text..."):
            docs = extract_and_split(uploaded_files)
        if not docs:
            st.error("No text extracted from the uploaded PDFs.")
            return

        if debug:
            st.sidebar.write(f"Total chunks: {len(docs)}")

        # Build or retrieve vector store
        vs = get_vector_store(docs)

        question = st.text_input("Ask a question about your PDFs:")
        if question:
            answer, sources = stream_answer(question, vs)
            st.success("Answer complete!")

            with st.expander("Sources & Context"):
                for doc in sources:
                    src = doc.metadata.get("source")
                    st.markdown(f"- **File:** {src}")
                    st.write(doc.page_content[:300] + "…")
    else:
        st.info("📁 Please upload at least one PDF to begin.")


if __name__ == "__main__":
    main()
