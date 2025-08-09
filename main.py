from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from openai import OpenAI
import os

print("OPEN_AI_API key inside app:", os.getenv("OPEN_AI_API"))

from detect_utils import damage_detection

# Get API key from environment
open_ai_key = os.getenv("OPEN_AI_API")
client = OpenAI(api_key=open_ai_key)

app = FastAPI()

class DamageRequest(BaseModel):
    damage_type: str

@app.get("/")
def read_root():
    return {"message": "Car Damage Assistant API is running."}

@app.post("/assist")
async def assist(request: DamageRequest):
    prompt = f"I have a car issue: {request.damage_type}. Can you tell me what to do until I reach a mechanic?"

    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful car repair assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=200
        )

        advice = response.choices[0].message.content
        return {
            "damage_type": request.damage_type,
            "advice": advice
        }
    except Exception as e:
        return {"error": str(e)}  

@app.post("/detect")
async def detect_endpoint(image: UploadFile = File(...)):
    try:
        print("Received /detect request")
        image_bytes = await image.read()
        labels = damage_detection(image_bytes)
        print("Detected labels:", labels)
        return {"detected_damage": labels}
    except Exception as e:
        print("Error in /detect:", e)
        return {"error": str(e)}

@app.post("/full-analysis")
async def full_analysis(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        labels = damage_detection(image_bytes)

        # If there isn't any damage detected then we should exit
        if not labels:
            return {"detected_damage": [], "analysis": []}

        # Build prompts for each detected damage
        analysis = []
        for label in labels:
            prompt = (
                f"The car has damage labeled as: {label}. "
                "Say something like 'looks like the car has (label issue)'"
                "what to tell a mechanic when they arrive, and what to do til they reach mechanic. "
                "Be concise and practical."
            )
            ai = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "You are a concise, practical car-repair assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=180
            )
            advice = ai.choices[0].message.content.strip()
            analysis.append({"damage": label, "advice": advice})

        return {"detected_damage": labels, "analysis": analysis}

    except Exception as e:
        return {"error": str(e)}