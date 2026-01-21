import os
import pinecone
from sentence_transformers import SentenceTransformer

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
INDEX_NAME = "jarvis-index"

if not PINECONE_API_KEY or not PINECONE_ENV:
    print("Please set PINECONE_API_KEY and PINECONE_ENV environment variables")
    exit(1)

pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

if INDEX_NAME not in pinecone.list_indexes():
    pinecone.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="cosine"
    )

index = pinecone.Index(INDEX_NAME)

model = SentenceTransformer("all-MiniLM-L6-v2")

docs = [
    "Our company follows strict data security policies.",
    "Confidential data must not be shared externally.",
    "AI assistants help improve productivity."
]

for i, doc in enumerate(docs):
    vector = model.encode(doc).tolist()
    index.upsert([(str(i), vector, {"text": doc})])

print("✅ Pinecone data inserted")
