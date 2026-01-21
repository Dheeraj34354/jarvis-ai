import os
from fastapi import FastAPI
from pydantic import BaseModel
from pinecone import Pinecone
import requests
from sentence_transformers import SentenceTransformer

# Pinecone config
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "jarvis-index"

if not PINECONE_API_KEY:
    raise ValueError("Please set PINECONE_API_KEY environment variable")

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

model = SentenceTransformer("all-MiniLM-L6-v2")

app = FastAPI()

class Query(BaseModel):
    question: str

def query_llama(prompt):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama2",
            "prompt": prompt,
            "stream": False
        }
    )
    return response.json()["response"]

@app.post("/chat")
def chat(query: Query):
    query_vector = model.encode(query.question).tolist()
    
    results = index.query(
        vector=query_vector,
        top_k=2,
        include_metadata=True
    )

    context = ""
    for match in results["matches"]:
        context += match["metadata"]["text"] + "\n"

    final_prompt = f"""
    Context:
    {context}

    Question:
    {query.question}

    Answer clearly:
    """

    answer = query_llama(final_prompt)
    return {"answer": answer}
