from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

# Load embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load FAISS vector database
db = FAISS.load_local(
    "vectorstore",
    embeddings,
    allow_dangerous_deserialization=True
)

# Load LLM
llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="liquid/lfm-2.5-1.2b-instruct:free"
)

# User Question
query = input("Ask a question: ")

# Retrieve context
docs_with_scores = db.similarity_search_with_score(query, k=5)

docs = []

for doc, score in docs_with_scores:
    if score < 1.0:
        docs.append(doc)

context = ""

for doc in docs:
    context += doc.page_content + "\n\n"

print("\n" + "="*50)
print("RETRIEVED CONTEXT")
print("="*50)
print(context[:1500])

# Generate Answer
answer_prompt = f"""
Answer ONLY using the provided context.

Context:
{context}

Question:
{query}
"""

answer = llm.invoke(answer_prompt).content

print("\n" + "="*50)
print("GENERATED ANSWER")
print("="*50)
print(answer)

# Evaluate Answer
evaluation_prompt = f"""
You are an AI evaluator.

Question:
{query}

Reference Context:
{context}

Generated Answer:
{answer}

Evaluate:

1. Correctness Score (0-10)
2. Faithfulness Score (0-10)
3. Hallucinations
4. Feedback

Return clearly.
"""

evaluation = llm.invoke(evaluation_prompt).content

print("\n" + "="*50)
print("EVALUATION")
print("="*50)
print(evaluation)