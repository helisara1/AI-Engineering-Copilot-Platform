from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

pdf_path = "uploads/Handbook of Data Structures and Applications [Mehta & Sahni 2004-10-28].pdf"

loader = PyPDFLoader(pdf_path)
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(docs)

print("Pages:", len(docs))
print("Chunks:", len(chunks))

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.from_documents(
    chunks,
    embeddings
)

db.save_local("vectorstore")

print("FAISS database saved successfully!")