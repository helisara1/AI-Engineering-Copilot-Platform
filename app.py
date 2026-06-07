import streamlit as st
import os

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_openai import ChatOpenAI

load_dotenv()

st.set_page_config(page_title="AI Engineering Copilot")

st.title("🤖 AI Engineering Copilot Platform")

# -------------------------------
# Upload PDF
# -------------------------------

uploaded_file = st.file_uploader(
    "Upload PDF",
    type="pdf"
)

if uploaded_file:

    os.makedirs("uploads", exist_ok=True)

    pdf_path = os.path.join(
        "uploads",
        uploaded_file.name
    )

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.read())

    st.success("PDF Uploaded Successfully")

    # -------------------------------
    # Build Vector Database
    # -------------------------------

    with st.spinner("Creating Vector Database..."):

        loader = PyPDFLoader(pdf_path)

        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        db = FAISS.from_documents(
            chunks,
            embeddings
        )

        db.save_local("vectorstore")

    st.success("Vector Database Created")

# -------------------------------
# Ask Question
# -------------------------------

question = st.text_input(
    "Ask a Question"
)

if question:

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.load_local(
        "vectorstore",
        embeddings,
        allow_dangerous_deserialization=True
    )

    docs = db.similarity_search(
        question,
        k=5
    )

    context = ""

    for doc in docs:
        context += doc.page_content + "\n\n"

    llm = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        model="liquid/lfm-2.5-1.2b-instruct:free"
    )

    # -------------------------------
    # Generate Answer
    # -------------------------------

    answer_prompt = f"""
    Answer ONLY using the provided context.

    Context:
    {context}

    Question:
    {question}
    """

    answer = llm.invoke(
        answer_prompt
    ).content

    st.subheader("Generated Answer")

    st.write(answer)

    # -------------------------------
    # Evaluate Answer
    # -------------------------------

    evaluation_prompt = f"""
    Question:
    {question}

    Reference Context:
    {context}

    Generated Answer:
    {answer}

    Evaluate:

    1. Correctness Score (0-10)
    2. Faithfulness Score (0-10)
    3. Hallucinations
    4. Feedback
    """

    evaluation = llm.invoke(
        evaluation_prompt
    ).content

    st.subheader("Evaluation")

    st.write(evaluation)

    # -------------------------------
    # Source Citations
    # -------------------------------

    st.subheader("Sources")

    for i, doc in enumerate(docs):

        with st.expander(
            f"Source {i+1}"
        ):
            st.write(
                doc.page_content
            )