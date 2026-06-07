from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    "vectorstore",
    embeddings,
    allow_dangerous_deserialization=True
)
query = input("Ask a question: ")

docs = db.similarity_search(query, k=3)

context = ""

for doc in docs:
    context += doc.page_content + "\n\n"

print("\nRetrieved Context:\n")
print(context[:3000])