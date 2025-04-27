import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

DIM = 384  # embedding size
index = None
id_map = {}  # maps FAISS ID to MongoDB ID

def init_faiss():
    global index, id_map
    index = faiss.IndexFlatL2(DIM)
    id_map = {}
    return index, id_map

def embed_text(text: str):
    embedding = model.encode([text])[0]
    return embedding.tolist()

def add_to_faiss(index, id_map, mongo_id: str, embedding: list):
    vec = np.array([embedding]).astype("float32")
    faiss_id = len(id_map)
    index.add(vec)
    id_map[faiss_id] = mongo_id

def query_faiss(index, query_text: str, top_k=5):
    embedding = model.encode([query_text]).astype("float32")
    D, I = index.search(embedding, top_k)
    return I[0]  # return FAISS indices of closest reviews