from dotenv import load_dotenv
import os

from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="liquid/lfm-2.5-1.2b-instruct:free"
)

question = input("Question: ")

student_answer = input("Student Answer: ")

context = input("Reference Context: ")

prompt = f"""
You are an evaluator.

Question:
{question}

Reference Context:
{context}

Student Answer:
{student_answer}

Evaluate:

1. Correctness (0-10)
2. Faithfulness (0-10)
3. Missing Concepts
4. Hallucinations
5. Final Feedback

Return in a structured format.
"""

response = llm.invoke(prompt)

print("\nEvaluation:\n")
print(response.content)