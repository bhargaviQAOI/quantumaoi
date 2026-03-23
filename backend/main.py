import json
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
import os
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI
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

CONTEXT_GUIDANCE = {
    "Email": "Format as a clear, professional email-style message with a natural opening and polished phrasing.",
    "LinkedIn": "Keep it concise, warm, and professional for a LinkedIn message or connection note.",
    "Job Application": "Sound thoughtful, polished, and tailored for a job application or candidate communication.",
    "Message": "Keep it natural, direct, and easy to read like a personal message.",
    "WhatsApp": "Make it casual, friendly, and easy to skim on mobile.",
    "Slack": "Make it concise, collaborative, and suitable for workplace chat.",
    "Instagram DM": "Keep it light, friendly, and conversational for social messaging.",
    "Cold Outreach": "Be concise, respectful, and persuasive without sounding pushy.",
    "Networking": "Sound warm, professional, and genuinely interested in building a connection.",
    "Follow-up": "Be polite, clear, and gently persistent.",
    "Recruiter Reply": "Sound professional, appreciative, and easy for a recruiter to respond to.",
    "Team Update": "Make it structured, clear, and collaborative for coworkers.",
    "Presentation Intro": "Make it polished, confident, and easy to deliver aloud.",
}

TONE_GUIDANCE = {
    "Professional": "Use polished, respectful, professional language.",
    "Confident": "Sound self-assured and clear without sounding arrogant.",
    "Polite": "Use courteous and considerate language.",
    "Executive": "Sound concise, strategic, and senior-level.",
    "Persuasive": "Make the message convincing while still sounding natural and respectful.",
    "Diplomatic": "Be tactful, balanced, and careful with sensitive wording.",
    "Direct": "Be clear, brief, and to the point.",
    "Friendly": "Sound approachable, warm, and easy to respond to.",
    "Warm": "Use kind, human, and encouraging phrasing.",
    "Conversational": "Sound natural and relaxed, like a real person speaking thoughtfully.",
    "Casual": "Keep the tone informal, simple, and comfortable.",
    "Modern Casual": "Use current, natural conversational language without overusing slang.",
    "Cheeky": "Add a light playful edge while keeping the message tasteful and appropriate.",
    "Playful": "Make it lively and upbeat without losing clarity.",
    "Crisp": "Make every sentence tight, efficient, and easy to scan.",
    "Human": "Prioritize natural, sincere language over corporate-sounding phrasing.",
    "Futuristic": "Sound sharp, innovative, and forward-looking while staying understandable.",
}

STYLE_GUIDANCE = {
    "default": "Keep the rewrite balanced and natural.",
    "confident": "Dial up clarity and confidence slightly while staying professional.",
    "short": "Make the wording shorter, sharper, and easier to scan.",
    "friendly": "Make it feel more human, warm, and approachable.",
}

SYSTEM_PROMPT = (
    "You rewrite real-world messages for clarity, tone, and professionalism. "
    "Be concise, natural, and practical."
)

class RewriteRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    context: Literal[
        "Email",
        "LinkedIn",
        "Job Application",
        "Message",
        "WhatsApp",
        "Slack",
        "Instagram DM",
        "Cold Outreach",
        "Networking",
        "Follow-up",
        "Recruiter Reply",
        "Team Update",
        "Presentation Intro",
    ]
    tone: Literal[
        "Professional",
        "Polite",
        "Executive",
        "Diplomatic",
        "Direct",
        "Crisp",
        "Confident",
        "Friendly",
        "Warm",
        "Conversational",
        "Human",
        "Casual",
        "Persuasive",
        "Playful",
        "Cheeky",
        "Modern Casual",
        "Futuristic",
    ]
    style: Literal["default", "confident", "short", "friendly"] = "default"

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Text cannot be empty.")
        return cleaned


def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not configured.")
    if api_key == "your_api_key_here":
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is still using the placeholder value.")
    return OpenAI(api_key=api_key)


def extract_versions(content: str) -> list[dict[str, str]]:
    cleaned_content = content.strip()

    if cleaned_content.startswith("```"):
        lines = cleaned_content.splitlines()
        if len(lines) >= 3:
            cleaned_content = "\n".join(lines[1:-1]).strip()

    try:
        payload = json.loads(cleaned_content)
    except json.JSONDecodeError:
        json_start = cleaned_content.find("{")
        json_end = cleaned_content.rfind("}")
        if json_start != -1 and json_end != -1 and json_end > json_start:
            try:
                payload = json.loads(cleaned_content[json_start:json_end + 1])
            except json.JSONDecodeError:
                payload = None
            if payload is not None:
                versions = payload.get("versions")
                if isinstance(versions, list) and len(versions) >= 2:
                    cleaned_content = cleaned_content[json_start:json_end + 1]
                else:
                    payload = None
        else:
            payload = None

        if payload is None:
            cleaned = cleaned_content
            if not cleaned:
                raise HTTPException(status_code=502, detail="The AI service returned an empty response.")
            return [
                {"label": "Version A", "text": cleaned},
                {"label": "Version B", "text": cleaned},
            ]

    versions = payload.get("versions")
    if not isinstance(versions, list) or len(versions) < 2:
        cleaned = content.strip()
        if not cleaned:
            raise HTTPException(status_code=502, detail="The AI service returned an empty response.")
        return [
            {"label": "Version A", "text": cleaned},
            {"label": "Version B", "text": cleaned},
        ]

    normalized_versions = []
    for index, item in enumerate(versions[:2]):
        text = item.get("text") if isinstance(item, dict) else None
        if not isinstance(text, str) or not text.strip():
            cleaned = content.strip()
            if not cleaned:
                raise HTTPException(status_code=502, detail="One of the rewrite versions was empty.")
            return [
                {"label": "Version A", "text": cleaned},
                {"label": "Version B", "text": cleaned},
            ]
        label = item.get("label") if isinstance(item, dict) else None
        normalized_versions.append(
            {
                "label": label if isinstance(label, str) and label.strip() else f"Version {'AB'[index]}",
                "text": text.strip(),
            }
        )

    return normalized_versions


@app.post("/rewrite")
def rewrite_text(request: RewriteRequest):
    context_guidance = CONTEXT_GUIDANCE.get(
        request.context,
        "Adapt the wording to fit the intended communication channel and audience.",
    )
    tone_guidance = TONE_GUIDANCE.get(
        request.tone,
        "Use a clear, natural, and appropriate tone for the message.",
    )
    style_guidance = STYLE_GUIDANCE.get(
        request.style,
        "Keep the rewrite balanced and natural.",
    )

    prompt = f"""
Return valid JSON only:
{{
  "versions": [
    {{"label": "Version A", "text": "..."}},
    {{"label": "Version B", "text": "..."}}
  ]
}}

Rewrite the message using:
- Context: {request.context}
- Tone: {request.tone}
- Style: {request.style}
- Context guidance: {context_guidance}
- Tone guidance: {tone_guidance}
- Style guidance: {style_guidance}

Rules:
- Keep the original meaning.
- Improve clarity, grammar, and flow.
- Sound natural, not robotic.
- Keep both versions concise.
- Version A = safest polished option.
- Version B = slightly different but still appropriate.

Message:
{request.text}
"""

    try:
        client = get_openai_client()
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.45,
            max_completion_tokens=500
        )
    except HTTPException:
        raise
    except APITimeoutError as exc:
        raise HTTPException(status_code=504, detail="The rewrite request timed out. Please try again.") from exc
    except APIConnectionError as exc:
        raise HTTPException(status_code=503, detail="Could not reach the AI service. Please try again.") from exc
    except APIStatusError as exc:
        raise HTTPException(status_code=502, detail="The AI service returned an unexpected error.") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Unexpected server error while rewriting text.") from exc

    rewritten_text = response.choices[0].message.content if response.choices else None
    if not rewritten_text:
        raise HTTPException(status_code=502, detail="The AI service returned an empty response.")

    versions = extract_versions(rewritten_text)

    return {
        "versions": versions
    }
