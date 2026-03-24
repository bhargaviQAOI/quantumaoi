import os
import re
from typing import Literal, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI
from pydantic import BaseModel, Field, field_validator

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_NAME = "gpt-4.1-mini"
MAX_COMPLETION_TOKENS = 260

MESSAGE_TYPE_GUIDANCE = {
    "Email": "Structure it cleanly, make the point early, and keep the flow polished.",
    "LinkedIn": "Keep it polished, concise, and warm without sounding salesy.",
    "Job Application": "Sound credible, thoughtful, and professionally sharp.",
    "Message": "Keep it natural, direct, and human.",
    "WhatsApp": "Keep it relaxed, easy to read, and conversational.",
    "Slack": "Keep it concise, useful, and workplace-natural.",
    "Instagram DM": "Keep it light, social, and natural.",
    "Cold Outreach": "Be respectful, sharp, and persuasive without sounding generic.",
    "Networking": "Sound warm, intentional, and genuinely interested.",
    "Follow-up": "Be clear, lightly persistent, and easy to reply to.",
    "Recruiter Reply": "Sound polished, appreciative, and efficient.",
    "Team Update": "Make it clear, structured, and action-oriented.",
    "Presentation Intro": "Make it smooth, confident, and easy to say out loud.",
}

TONE_GUIDANCE = {
    "Professional": "Sound polished, confident, and clear.",
    "Confident": "Be decisive and assured without sounding aggressive.",
    "Polite": "Stay courteous without becoming overly soft or wordy.",
    "Executive": "Keep it crisp, high-signal, and senior.",
    "Persuasive": "Make the case clearly and naturally.",
    "Diplomatic": "Be tactful, balanced, and measured.",
    "Direct": "Say the point plainly and efficiently.",
    "Friendly": "Sound warm, easygoing, and natural.",
    "Warm": "Sound thoughtful, human, and approachable.",
    "Conversational": "Write like a smart person speaking naturally.",
    "Human": "Prioritize natural phrasing over polished corporate phrasing.",
    "Casual": "Keep it relaxed, simple, and natural.",
    "Playful": "Add light personality without losing clarity.",
    "Cheeky": "Add light edge while staying tasteful.",
    "Modern Casual": "Use current, natural language without forcing slang.",
    "Crisp": "Tighten every sentence and remove drag.",
    "Futuristic": "Keep it sharp and forward-looking without sounding unnatural.",
}

STYLE_GUIDANCE = {
    "default": "Keep the rewrite balanced, strong, and natural.",
    "confident": "Push it slightly more assertive and decisive.",
    "short": "Make it tighter and more economical.",
    "friendly": "Make it warmer and more approachable.",
}

VOICE_GUIDANCE = (
    "Write with a clear, confident, slightly direct voice. "
    "Keep it natural, modern, and human. "
    "Prefer clean wording over padded or overly careful phrasing."
)

POLISH_GUIDANCE = "\n".join(
    [
        "- Add a final light polish before answering: tighten wording, remove softness, and smooth the flow.",
        "- Keep the polish subtle. Do not do a full second rewrite.",
        "- Make the ending feel intentional and context-aware, not generic.",
        "- Trim one or two unnecessary words when possible without changing meaning.",
    ]
)

VARIATION_GUIDANCE = (
    "Allow slight variation in sentence openings and phrasing so outputs do not all sound templated, "
    "while keeping the overall voice consistent."
)

ANTI_AI_GUIDANCE = "\n".join(
    [
        "- Avoid generic AI phrasing and empty courtesy lines.",
        "- Do not use phrases like 'I hope you are doing well', 'I just wanted to', or 'kindly'.",
        "- Avoid stiff transitions and overly formal connectors unless the message truly requires them.",
        "- Prefer modern, natural communication over template-like professionalism.",
        "- Avoid generic endings like 'let me know' or 'thanks in advance' unless they genuinely fit.",
    ]
)

ACTION_GUIDANCE = {
    "shorten": "Make it tighter, sharper, and more intentional rather than simply shorter.",
    "friendlier": "Make it warmer and easier to connect with while keeping it natural and unsentimental.",
    "stronger": "Make it more direct, confident, and intentional without sounding aggressive.",
}

SYSTEM_PROMPT = """
You are a premium AI rewrite engine.
Your job is not to lightly edit. Your job is to materially improve writing so it feels sharper, smarter, cleaner, and more human-written.
You should sound like an opinionated writing assistant with strong taste, not a generic paraphraser.

Principles:
- Make the rewrite noticeably better than the source.
- Do not sound robotic, generic, timid, or over-polite.
- Prefer strong phrasing, clean structure, and natural rhythm.
- Rewrite deeply when needed: reorder ideas, split or combine sentences, and replace weak phrasing.
- Avoid mirroring the source sentence-by-sentence.
- Keep the original intent, but upgrade the writing.
- Favor clear, confident, slightly direct language.
- Remove weak phrasing, filler, and default corporate politeness.
- Apply a subtle final polish so the result feels refined, sharp, and occasionally impressive.
- Keep some controlled variation in phrasing so outputs do not feel mass-produced.

Before finalizing, silently check:
1. Is this clearer than the source?
2. Is it more concise or more efficient?
3. Does it sound like a smart human wrote it?
If not, refine once before answering.

Always respond in exactly this format:
TYPE: <message type>
OUTPUT: <final text>
""".strip()


class RewriteRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    action: Optional[Literal["shorten", "friendlier", "stronger"]] = None
    previous_output: Optional[str] = Field(default=None, max_length=5000)
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
    ] = "Email"
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
    ] = "Professional"
    style: Literal["default", "confident", "short", "friendly"] = "default"

    @field_validator("text", "previous_output")
    @classmethod
    def validate_optional_text(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
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


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def word_count(text: str) -> int:
    return len(re.findall(r"\w+", text))


def detect_message_type(text: str, context: Optional[str]) -> str:
    lowered = text.lower()

    if any(token in lowered for token in ("subject:", "dear ", "best,", "kind regards", "sincerely")):
        return "Email"
    if any(token in lowered for token in ("follow up", "following up", "checking in", "circling back")):
        return "Follow-up"
    if any(token in lowered for token in ("application", "resume", "interview", "hiring", "role")):
        return "Job Application"
    if any(token in lowered for token in ("slack", "thread", "channel", "standup", "blocker")):
        return "Slack"
    if any(token in lowered for token in ("linkedin", "connection request")):
        return "LinkedIn"
    if any(token in lowered for token in ("hey", "haha", "lol", "wanna", "gonna", "btw")):
        return "WhatsApp"
    if context and context != "Email":
        return context
    if "\n" in text and word_count(text) > 35:
        return "Email"
    return context or "Message"


def normalize_action(action: Optional[str], style: str) -> Optional[str]:
    if action:
        return action

    style_to_action = {
        "short": "shorten",
        "friendly": "friendlier",
        "confident": "stronger",
    }
    return style_to_action.get(style)


def build_quality_rules(message_type: str) -> str:
    return "\n".join(
        [
            f"- Treat this as a {message_type} and shape the writing accordingly.",
            "- The output must feel like a meaningful upgrade, not a light edit.",
            "- Improve structure, flow, and clarity before polishing word choice.",
            "- Use stronger, more intentional phrasing.",
            "- Bias toward shorter, cleaner, sharper wording.",
            "- Use a clear, confident, slightly direct voice.",
            "- Vary sentence length naturally when it helps the rhythm.",
            "- Improve rhythm and readability, not just correctness.",
            "- Remove filler, hedging, repetition, and generic AI-sounding wording.",
            "- Simplify clunky phrasing, combine ideas when the writing feels choppy, and split ideas when a sentence gets heavy.",
            "- Preserve meaning, but do not preserve weak phrasing.",
            "- Keep it concise and high-signal.",
            "- Replace weak phrasing with stronger alternatives when it improves the line.",
            VARIATION_GUIDANCE,
            POLISH_GUIDANCE,
            ANTI_AI_GUIDANCE,
            "- Return only the required TYPE/OUTPUT format.",
        ]
    )


def build_rewrite_prompt(request: RewriteRequest, message_type: str) -> str:
    message_guidance = MESSAGE_TYPE_GUIDANCE.get(
        message_type,
        "Adapt the message naturally to its use case.",
    )
    tone_guidance = TONE_GUIDANCE.get(
        request.tone,
        "Keep the tone clear and natural.",
    )
    style_guidance = STYLE_GUIDANCE.get(
        request.style,
        STYLE_GUIDANCE["default"],
    )

    return f"""
Rewrite this message so it feels premium and clearly stronger than the original.

Detected message type: {message_type}
Message-type guidance: {message_guidance}
Tone guidance: {tone_guidance}
Style guidance: {style_guidance}
Voice guidance: {VOICE_GUIDANCE}
Variation guidance: {VARIATION_GUIDANCE}

Rewrite rules:
{build_quality_rules(message_type)}

Source message:
{request.text}
""".strip()


def build_action_prompt(request: RewriteRequest, message_type: str, action: str) -> str:
    message_guidance = MESSAGE_TYPE_GUIDANCE.get(
        message_type,
        "Adapt the message naturally to its use case.",
    )
    tone_guidance = TONE_GUIDANCE.get(
        request.tone,
        "Keep the tone clear and natural.",
    )
    action_guidance = ACTION_GUIDANCE[action]

    return f"""
Refine an already strong rewrite.

Detected message type: {message_type}
Message-type guidance: {message_guidance}
Tone guidance: {tone_guidance}
Requested action: {action}
Action guidance: {action_guidance}
Voice guidance: {VOICE_GUIDANCE}
Variation guidance: {VARIATION_GUIDANCE}

Rules:
{build_quality_rules(message_type)}
- Refine the previous output instead of rewriting from scratch.
- Keep the voice, intent, and general direction of the previous output.
- Improve quality further. Do not make it flatter, weaker, or more generic.
- Use the original text only to preserve meaning.

Original text:
{request.text}

Previous output:
{request.previous_output}
""".strip()


def build_prompt(request: RewriteRequest) -> tuple[str, str, str]:
    message_type = detect_message_type(request.text, request.context)
    action = normalize_action(request.action, request.style)

    if action and request.previous_output:
        prompt = build_action_prompt(request, message_type, action)
        source_for_similarity = request.previous_output
        return prompt, message_type, source_for_similarity

    prompt = build_rewrite_prompt(request, message_type)
    return prompt, message_type, request.text


def strip_code_fences(content: str) -> str:
    cleaned = content.strip()
    if cleaned.startswith("```") and cleaned.endswith("```"):
        lines = cleaned.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    return cleaned


def parse_model_output(content: str, fallback_type: str) -> dict[str, str]:
    cleaned = strip_code_fences(content)

    type_match = re.search(r"^\s*TYPE:\s*(.+?)\s*$", cleaned, re.IGNORECASE | re.MULTILINE)
    output_match = re.search(r"^\s*OUTPUT:\s*(.*)$", cleaned, re.IGNORECASE | re.MULTILINE | re.DOTALL)

    parsed_type = fallback_type
    parsed_output = ""

    if type_match:
        parsed_type = normalize_whitespace(type_match.group(1))

    if output_match:
        parsed_output = output_match.group(1).strip()
    else:
        lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
        remaining_lines = [line for line in lines if not line.lower().startswith("type:")]
        if remaining_lines:
            first_line = remaining_lines[0]
            if first_line.lower().startswith("output:"):
                first_line = first_line[7:].strip()
                parsed_output = "\n".join([first_line, *remaining_lines[1:]]).strip()
            else:
                parsed_output = "\n".join(
                    line for line in remaining_lines if not line.lower().startswith("output:")
                ).strip()

    parsed_output = parsed_output.strip().strip("\"'`")
    if not parsed_output:
        parsed_output = cleaned
        parsed_output = re.sub(r"^\s*TYPE:\s*.*$", "", parsed_output, flags=re.IGNORECASE | re.MULTILINE)
        parsed_output = re.sub(r"^\s*OUTPUT:\s*", "", parsed_output, flags=re.IGNORECASE | re.MULTILINE)
        parsed_output = parsed_output.strip().strip("\"'`")

    if not parsed_output:
        raise HTTPException(status_code=502, detail="The AI service returned an empty rewrite.")

    return {
        "type": parsed_type or fallback_type,
        "output": parsed_output,
    }


def lexical_overlap_ratio(source_text: str, rewritten_text: str) -> float:
    source_tokens = re.findall(r"\w+", source_text.lower())
    rewritten_tokens = re.findall(r"\w+", rewritten_text.lower())
    if not source_tokens or not rewritten_tokens:
        return 0.0

    source_token_set = set(source_tokens)
    shared_tokens = sum(1 for token in rewritten_tokens if token in source_token_set)
    return shared_tokens / max(len(rewritten_tokens), 1)


def needs_stronger_retry(source_text: str, rewritten_text: str) -> bool:
    normalized_source = normalize_whitespace(source_text).lower()
    normalized_output = normalize_whitespace(rewritten_text).lower()

    if not normalized_output:
        return True
    if normalized_source == normalized_output:
        return True
    if lexical_overlap_ratio(source_text, rewritten_text) > 0.82:
        return True

    source_words = word_count(source_text)
    output_words = word_count(rewritten_text)
    if source_words >= 12 and output_words >= source_words * 1.3:
        return True

    return False


def build_retry_prompt(prompt: str) -> str:
    return (
        f"{prompt}\n\n"
        "Refine once more.\n"
        "The previous answer was still too safe or too close to the source.\n"
        "Make it more concise and direct.\n"
        "Remove weak phrasing and any generic or overly polite language.\n"
        "Improve flow, rhythm, and sentence quality.\n"
        "Tighten the ending so it feels intentional rather than default.\n"
        "Restructure where useful instead of paraphrasing line by line.\n"
        "Do not become longer unless length is clearly necessary for clarity."
    )


def apply_micro_polish(text: str) -> str:
    polished = text.strip()

    substitutions = (
        (r"\bI wanted to check\b", "Checking"),
        (r"\bI wanted to follow up\b", "Following up"),
        (r"\bI just wanted to\b", ""),
        (r"\bkindly\b", ""),
        (r"\bplease let me know\b", "Let me know"),
        (r"\bthanks in advance\b", "Thanks"),
        (r"\bit would be great if\b", "It'd be great if"),
    )

    for pattern, replacement in substitutions:
        polished = re.sub(pattern, replacement, polished, flags=re.IGNORECASE)

    polished = re.sub(r"\s+", " ", polished)
    polished = re.sub(r"\s+([,.;:!?])", r"\1", polished)
    polished = re.sub(r"\s{2,}", " ", polished)
    return polished.strip()


def create_messages(prompt: str, retry: bool = False) -> list[dict[str, str]]:
    user_prompt = build_retry_prompt(prompt) if retry else prompt
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def request_rewrite(
    client: OpenAI,
    prompt: str,
    source_text: str,
    fallback_type: str,
) -> dict[str, str]:
    for retry in (False, True):
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=create_messages(prompt, retry=retry),
            temperature=0.5 if not retry else 0.6,
            max_completion_tokens=MAX_COMPLETION_TOKENS,
        )

        content = response.choices[0].message.content if response.choices else None
        if not content:
            raise HTTPException(status_code=502, detail="The AI service returned an empty response.")

        parsed = parse_model_output(content, fallback_type)
        parsed["output"] = apply_micro_polish(parsed["output"])
        if retry or not needs_stronger_retry(source_text, parsed["output"]):
            return parsed

    raise HTTPException(status_code=502, detail="The AI service could not produce a strong enough rewrite.")


@app.post("/rewrite")
def rewrite_text(request: RewriteRequest):
    prompt, fallback_type, source_for_similarity = build_prompt(request)

    try:
        client = get_openai_client()
        parsed = request_rewrite(
            client=client,
            prompt=prompt,
            source_text=source_for_similarity,
            fallback_type=fallback_type,
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

    return {
        "versions": [
            {
                "label": "Improved",
                "text": parsed["output"],
            }
        ],
        "rewritten_text": parsed["output"],
        "message_type": parsed["type"],
    }
