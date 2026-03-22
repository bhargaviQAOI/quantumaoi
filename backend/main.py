from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all for now
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class RewriteRequest(BaseModel):
    text: str
    context: str
    tone: str


@app.post("/rewrite")
def rewrite_text(request: RewriteRequest):
    prompt = f"""
You are a professional communication coach.

Rewrite the following text based on:
Context: {request.context}
Tone: {request.tone}

Make it clear, confident, and professional. Improve soft skills.

Text:
{request.text}
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "You improve communication."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )

    return {
        "rewritten_text": response.choices[0].message.content
    }