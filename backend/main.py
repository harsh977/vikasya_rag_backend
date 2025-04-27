from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from pymongo import MongoClient
import os
import dotenv
from gemini import generate_response, generate_summary
from rag import embed_text, add_to_faiss, query_faiss, init_faiss
import uvicorn  # ADDED

dotenv.load_dotenv()
app = FastAPI()

# --- CORS Configuration ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict to your frontend domain later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- MongoDB Connection ---
client = MongoClient(os.getenv("MONGO_URI"))
db = client["public_services_db"]
feedback_collection = db["feedbacks"]

# --- FAISS Index ---
index, id_map = init_faiss()

# --- Pydantic Model ---
class Feedback(BaseModel):
    district_name: str
    service_type: str
    user_feedback: str

# --- Submit Feedback Route ---
@app.post("/submit_feedback/")
async def submit_feedback(feedback: Feedback):
    try:
        embedding = embed_text(feedback.user_feedback)
        response = generate_response(feedback.user_feedback, embedding)

        feedback_data = {
            "district_name": feedback.district_name,
            "service_type": feedback.service_type,
            "user_feedback": feedback.user_feedback,
            "response_text": response,
            "embedding": embedding
        }
        result = feedback_collection.insert_one(feedback_data)

        add_to_faiss(index, id_map, str(result.inserted_id), embedding)

        return {"response": response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Generate Summary Route ---
@app.get("/summary/{district_name}/{service_type}")
async def get_summary(district_name: str, service_type: str):
    feedbacks = list(feedback_collection.find({
        "district_name": district_name,
        "service_type": service_type
    }))
    if not feedbacks:
        raise HTTPException(status_code=404, detail="No feedback found for this district and service.")

    texts = [f["user_feedback"] for f in feedbacks]
    embeddings = [f["embedding"] for f in feedbacks]
    summary = generate_summary(texts, embeddings)

    return {"summary": summary}

# --- Welcome Route ---
@app.get("/")
async def root():
    return {"message": "Welcome to the Public Services Feedback API 🚍🛣️"}

# --- Uvicorn Start for Render ---
