from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from datetime import date, datetime
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util

app = FastAPI()


model = SentenceTransformer("all-MiniLM-L6-v2")

class MedicineQuery(BaseModel):
    query: str

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, replace '*' with your frontend IP/domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get('/')
def read_root():
    return {"message" : " hello from the fastapi"}

@app.post("/search_medicine")
async def search_medicine(query: MedicineQuery):
    try:
        query_embedding = model.encode(query.query)
        
        similarities = util.cos_sim(query_embedding, embeddings)[0]
        
        best_match_idx = similarities.argmax().item()
        
        medicine_details = df.iloc[best_match_idx].to_dict()

        return {
            "status": "success",
            "data": medicine_details
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")
