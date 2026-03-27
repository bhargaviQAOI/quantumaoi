import os
import re
import json
from datetime import date
from typing import Literal, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI
from pydantic import BaseModel, Field, field_validator
import requests

load_dotenv(override=False)
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"), override=False)

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
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL_NAME = "openchat/openchat-3.5"
OPENROUTER_ERROR_RESPONSES = {"AI service error", "Server error"}
FREE_DAILY_LIMIT = 10
FREE_LIMIT_MESSAGE = "You’ve used today’s free messages.\nCome back tomorrow — or unlock unlimited access ✨"
LAST_FREE_MESSAGE_NOTICE = "That was your last free message today.\nUnlock unlimited to keep going ✨"
INVALID_INPUT_RESPONSES = (
    "That doesn't look like a real message. Try pasting something you'd send.",
    "Looks like placeholder text — paste a real message and I'll clean it up.",
    "That doesn't look like a real message. Try again with something you'd send.",
)
daily_usage_by_ip: dict[tuple[str, str], int] = {}

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
    "simple": "Use basic words, short sentences, and make the meaning easy to understand.",
    "polite": "Make it respectful, friendly, and easy to receive without sounding stiff.",
    "strong": "Make it more confident, direct, and clear without sounding aggressive.",
    "technical": "Make it more precise, specific, and professionally technical while staying clear and readable.",
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
    "simple": "Rewrite with basic words, short sentences, and very clear meaning.",
    "polite": "Make it more respectful and friendly while keeping it natural and concise.",
    "strong": "Make it more confident and direct while keeping it natural and readable.",
    "technical": "Increase precision and specificity, use professional terminology where it helps, and keep the message concise and clear.",
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
- Rewrite for context and intent, not literal translation.
- Do NOT preserve awkward phrases. Rewrite naturally like a native speaker. Keep it concise and clear.
- Do NOT preserve awkward or unnatural phrases. Rewrite the message as a native speaker would naturally say it.
- Do not carry over unusual, loaded, or awkward source words unless they clearly fit the situation.
- Prefer clarity, simplicity, and natural tone over literal translation.
- Prefer natural phrasing over direct word-for-word conversion.
- Keep the original intent, but upgrade the writing.
- Favor clear, confident, slightly direct language.
- Remove weak phrasing, filler, and default corporate politeness.
- Keep the message concise. Do not add unnecessary sentences.
- Apply a subtle final polish so the result feels refined, sharp, and occasionally impressive.
- Keep some controlled variation in phrasing so outputs do not feel mass-produced.

Before finalizing, silently check:
1. Is this clearer than the source?
2. Is it more concise or more efficient?
3. Does it sound like a smart human wrote it?
If not, refine once before answering.

If the input is mostly symbols, placeholder text, or not a real message, respond like this instead:
TYPE: invalid_input
OUTPUT: a short, friendly note asking for a real message to rewrite.

Always respond in exactly this format:
TYPE: <message type>
OUTPUT: <final text>

In OUTPUT, return only the improved message. Do not include explanations or extra text.
""".strip()


class RewriteRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    action: Optional[Literal["shorten", "friendlier", "stronger", "simple", "polite", "strong", "technical"]] = None
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
    style: Literal["default", "confident", "short", "friendly", "simple", "polite", "strong", "technical"] = "default"

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


def get_openrouter_api_key() -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key and openai_api_key.startswith("sk-or-v1-"):
            api_key = openai_api_key
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY is not configured.")
    if api_key == "your_api_key_here":
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY is still using the placeholder value.")
    return api_key


def get_client_ip(request: Request) -> str:
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        first_ip = forwarded_for.split(",")[0].strip()
        if first_ip:
            return first_ip
    if request.client and request.client.host:
        return request.client.host
    return "anonymous"


def get_daily_usage_count(client_ip: str) -> int:
    usage_key = (client_ip, date.today().isoformat())
    return daily_usage_by_ip.get(usage_key, 0)


def consume_free_usage(client_ip: str) -> tuple[bool, int]:
    usage_key = (client_ip, date.today().isoformat())
    current_count = daily_usage_by_ip.get(usage_key, 0)
    if current_count >= FREE_DAILY_LIMIT:
        return False, current_count
    updated_count = current_count + 1
    daily_usage_by_ip[usage_key] = updated_count
    return True, updated_count


def build_rewrite_response(
    output_text: str,
    message_type: str = "Message",
    is_limit: bool = False,
    is_premium: bool = False,
    usage_count: Optional[int] = None,
    usage_limit: Optional[int] = None,
    usage_notice: Optional[str] = None,
) -> dict[str, object]:
    return {
        "text": output_text,
        "is_limit": is_limit,
        "is_premium": is_premium,
        "usage": {
            "used": usage_count,
            "limit": usage_limit,
        },
        "output": output_text,
        "versions": [
            {
                "label": "Improved",
                "text": output_text,
            }
        ],
        "rewritten_text": output_text,
        "message_type": message_type,
        "usage_count": usage_count,
        "usage_limit": usage_limit,
        "usage_notice": usage_notice,
    }


def looks_invalid_input(text: str) -> bool:
    alpha_tokens = re.findall(r"[A-Za-z]+(?:['-][A-Za-z]+)?", text)
    alnum_chars = re.findall(r"[A-Za-z0-9]", text)
    symbol_chars = re.findall(r"[^\w\s]", text)

    if not alnum_chars:
        return True
    if symbol_chars and len(symbol_chars) > len(alnum_chars):
        return True
    if not alpha_tokens:
        return True
    if len(alpha_tokens) < 3 and sum(len(token) for token in alpha_tokens) < 5:
        return True

    return False


def build_invalid_input_response() -> dict[str, object]:
    return build_rewrite_response(
        INVALID_INPUT_RESPONSES[0],
        message_type="invalid_input",
    )


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


def detect_intent(text: str) -> str:
    stripped = text.strip()
    lowered = stripped.lower()

    if "?" in stripped or lowered.startswith(
        ("can you", "could you", "would you", "are you", "is there", "do you", "did you", "have you")
    ):
        return "question"

    if lowered.startswith(
        ("please", "can you", "could you", "would you", "let's", "pls", "do ", "send ", "share ")
    ):
        return "request"

    return "statement"


def normalize_action(action: Optional[str], style: str) -> Optional[str]:
    if action:
        return action

    style_to_action = {
        "short": "shorten",
        "friendly": "friendlier",
        "confident": "stronger",
        "simple": "simple",
        "polite": "polite",
        "strong": "strong",
        "technical": "technical",
    }
    return style_to_action.get(style)


def build_quality_rules(message_type: str, intent: str) -> str:
    return "\n".join(
        [
            f"- Treat this as a {message_type} and shape the writing accordingly.",
            f"- Preserve the intent as a {intent}; do not turn a question into a statement or a request into a demand.",
            "- The output must feel like a meaningful upgrade, not a light edit.",
            "- Improve structure, flow, and clarity before polishing word choice.",
            "- Use stronger, more intentional phrasing.",
            "- Bias toward shorter, cleaner, sharper wording.",
            "- Use a clear, confident, slightly direct voice.",
            "- Vary sentence length naturally when it helps the rhythm.",
            "- Improve rhythm and readability, not just correctness.",
            "- Remove filler, hedging, repetition, and generic AI-sounding wording.",
            "- Simplify clunky phrasing, combine ideas when the writing feels choppy, and split ideas when a sentence gets heavy.",
            "- Rewrite from meaning and context, not from literal word carryover.",
            "- Do NOT preserve awkward or unnatural phrases. Rewrite the message as a native speaker would naturally say it.",
            "- Avoid preserving unusual or culturally loaded words unless they are clearly appropriate in natural English.",
            "- Prefer clarity, simplicity, and natural tone over literal translation.",
            "- Preserve meaning, but do not preserve weak phrasing.",
            "- Keep it concise and high-signal.",
            "- Keep the message concise. Do not add unnecessary sentences.",
            "- Replace weak phrasing with stronger alternatives when it improves the line.",
            "- Use more precise and domain-appropriate terminology when the message is technical or work-related.",
            "- Keep technical rewrites readable; do not add jargon just to sound smarter.",
            VARIATION_GUIDANCE,
            POLISH_GUIDANCE,
            ANTI_AI_GUIDANCE,
            "- In OUTPUT, return only the improved message. Do not include explanations or extra text.",
            "- Return only the required TYPE/OUTPUT format.",
        ]
    )


def build_rewrite_prompt(request: RewriteRequest, message_type: str) -> str:
    intent = detect_intent(request.text)
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
Detected intent: {intent}
Message-type guidance: {message_guidance}
Tone guidance: {tone_guidance}
Style guidance: {style_guidance}
Voice guidance: {VOICE_GUIDANCE}
Variation guidance: {VARIATION_GUIDANCE}

Rewrite rules:
{build_quality_rules(message_type, intent)}

Source message:
{request.text}
""".strip()


def build_action_prompt(request: RewriteRequest, message_type: str, action: str) -> str:
    intent = detect_intent(request.text)
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
Detected intent: {intent}
Message-type guidance: {message_guidance}
Tone guidance: {tone_guidance}
Requested action: {action}
Action guidance: {action_guidance}
Voice guidance: {VOICE_GUIDANCE}
Variation guidance: {VARIATION_GUIDANCE}

Rules:
{build_quality_rules(message_type, intent)}
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
        "Rewrite any awkward phrasing the way a native speaker would naturally say it.\n"
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


def rewrite_with_openai(text: str) -> str:
    client = get_openai_client()
    parsed = request_rewrite(
        client=client,
        prompt=text,
        source_text=text,
        fallback_type="Message",
    )
    return parsed["output"]


def rewrite_with_openrouter(text: str) -> str:
    try:
        api_key = get_openrouter_api_key()
        payload = {
            "model": OPENROUTER_MODEL_NAME,
            "messages": [
                {
                    "role": "user",
                    "content": text,
                }
            ],
            "temperature": 0.7,
            "max_tokens": 150,
        }

        response = requests.post(
            OPENROUTER_API_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=10,
        )
        data = response.json()
        print("==== OPENROUTER RESPONSE ====")
        print(data)
        print("==== END RESPONSE ====")

        if "error" in data:
            print("OpenRouter error:", data)
            print("Provider fallback: OpenAI")
            return rewrite_with_openai(text)

        if not response.ok:
            print("OpenRouter error:", data)
            print("Provider fallback: OpenAI")
            return rewrite_with_openai(text)

        if "choices" not in data:
            print("OpenRouter error:", data)
            print("Provider fallback: OpenAI")
            return rewrite_with_openai(text)

        choices = data.get("choices") or []
        if not choices:
            print("OpenRouter error:", data)
            print("Provider fallback: OpenAI")
            return rewrite_with_openai(text)

        message = choices[0].get("message", {})
        content = message.get("content")
        if not content:
            print("OpenRouter error:", data)
            print("Provider fallback: OpenAI")
            return rewrite_with_openai(text)

        return content
    except Exception as e:
        print("Error:", e)
        print("Provider fallback: OpenAI")
        return rewrite_with_openai(text)


def request_openrouter_rewrite(
    prompt: str,
    source_text: str,
    fallback_type: str,
) -> Optional[dict[str, str]]:
    for retry in (False, True):
        content = rewrite_with_openrouter(build_retry_prompt(prompt) if retry else prompt)
        if content in OPENROUTER_ERROR_RESPONSES:
            return None

        try:
            parsed = parse_model_output(content, fallback_type)
        except HTTPException as exc:
            print("OpenRouter parse error:", exc.detail)
            return None

        parsed["output"] = apply_micro_polish(parsed["output"])
        if retry or not needs_stronger_retry(source_text, parsed["output"]):
            return parsed

    return None


@app.post("/rewrite")
def rewrite_text(request: RewriteRequest, http_request: Request):
    if looks_invalid_input(request.text):
        return build_invalid_input_response()

    prompt, fallback_type, source_for_similarity = build_prompt(request)
    is_premium = (
        http_request.query_params.get("premium") == "true"
        or http_request.headers.get("X-User-Type") == "premium"
    )
    usage_count = None
    usage_notice = None
    client_ip = get_client_ip(http_request)
    if not is_premium:
        has_usage_remaining, usage_count = consume_free_usage(client_ip)
        print(f"[USAGE] {usage_count}/{FREE_DAILY_LIMIT} for {client_ip}")
        if not has_usage_remaining:
            print("Usage limit reached:", client_ip)
            return build_rewrite_response(
                FREE_LIMIT_MESSAGE,
                message_type="Limit Reached",
                is_limit=True,
                is_premium=False,
                usage_count=usage_count,
                usage_limit=FREE_DAILY_LIMIT,
            )
        if usage_count == FREE_DAILY_LIMIT:
            usage_notice = LAST_FREE_MESSAGE_NOTICE

    try:
        if is_premium:
            print("Provider: OpenAI")
            client = get_openai_client()
            parsed = request_rewrite(
                client=client,
                prompt=prompt,
                source_text=source_for_similarity,
                fallback_type=fallback_type,
            )
        else:
            print("Provider: OpenRouter")
            try:
                parsed = request_openrouter_rewrite(
                    prompt=prompt,
                    source_text=source_for_similarity,
                    fallback_type=fallback_type,
                )
            except Exception as exc:
                print("Provider fallback: OpenRouter failed", exc)
                parsed = None

            if parsed is None:
                print("Provider fallback: OpenAI")
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

    return build_rewrite_response(
        parsed["output"],
        message_type=parsed["type"],
        is_limit=False,
        is_premium=is_premium,
        usage_count=usage_count,
        usage_limit=FREE_DAILY_LIMIT if not is_premium else None,
        usage_notice=usage_notice,
    )
