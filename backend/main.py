from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pymongo import MongoClient
from bson import ObjectId
import os
import dotenv
import traceback
from gemini import generate_response, generate_summary
from rag import VectorDB, embed_text

# Initialize environment and app
dotenv.load_dotenv()
app = FastAPI()

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database setup
client = MongoClient(os.getenv("MONGO_URI"))
db = client["public_services_db"]
feedback_collection = db["feedbacks"]

# Initialize VectorDB
vector_db = VectorDB()

# Pydantic model
class Feedback(BaseModel):
    district_name: str
    service_type: str
    user_feedback: str

# Routes
@app.post("/submit_feedback/")
async def submit_feedback(feedback: Feedback):
    try:
        # Generate embedding
        embedding = embed_text(feedback.user_feedback)
        
        # Find similar feedbacks
        similar_ids = vector_db.query_vectors(embedding)
        similar_feedbacks = [
            doc["user_feedback"] 
            for doc in feedback_collection.find({"_id": {"$in": [ObjectId(id) for id in similar_ids]}})
        ]
        
        # Generate response
        response = generate_response(feedback.user_feedback, similar_feedbacks)
        
        # Store in MongoDB
        feedback_data = {
            "district_name": feedback.district_name,
            "service_type": feedback.service_type,
            "user_feedback": feedback.user_feedback,
            "response_text": response,
            "embedding": embedding
        }
        result = feedback_collection.insert_one(feedback_data)
        
        # Update VectorDB
        vector_db.add_vector(str(result.inserted_id), embedding)
        
        return {"response": response}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, detail=f"Feedback processing failed: {str(e)}")

@app.get("/summary/{district_name}/{service_type}")
async def get_summary(district_name: str, service_type: str):
    try:
        # Get main feedbacks
        main_feedbacks = list(feedback_collection.find({
            "district_name": district_name,
            "service_type": service_type
        }))
        
        if not main_feedbacks:
            raise HTTPException(404, detail="No feedbacks found")
        
        # Get similar feedbacks
        all_embeddings = [doc["embedding"] for doc in main_feedbacks]
        similar_ids = []
        for emb in all_embeddings:
            similar_ids.extend(vector_db.query_vectors(emb, top_k=2))
        
        # Remove duplicates
        similar_ids = list(dict.fromkeys(similar_ids))
        
        similar_feedbacks = [
            doc["user_feedback"] 
            for doc in feedback_collection.find({"_id": {"$in": [ObjectId(id) for id in similar_ids]}})
        ]
        
        # Generate summary
        summary = generate_summary(
            [doc["user_feedback"] for doc in main_feedbacks],
            similar_feedbacks
        )
        
        return {"summary": summary}

    except HTTPException as he:
        raise he
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, detail=f"Summary generation failed: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Public Services Feedback Analysis API"}