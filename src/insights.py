"""Gemini API call — the only place in this pipeline that talks to an LLM.

Hard constraint (see CLAUDE.md): Gemini only ever narrates precomputed facts
from src/retrieval.py. It never receives raw invoice rows, and it is
explicitly instructed not to perform any calculation itself — every number
in its output must already exist in the facts JSON it's given. This is
enforced structurally, not just by asking nicely: the facts block is the
only source of numbers in the request at all.

Called via plain REST (no SDK) per the project's stack decision.
"""

import json
import os
import time

import requests
from dotenv import load_dotenv

load_dotenv()

# Tried in order. Flash-Lite first: the smallest current model is plenty for
# narrating a handful of facts. The free tier is frequently "busy" (503) or
# slow, so each model gets one attempt and the next one is tried on any
# transient failure. GEMINI_MODEL overrides the first choice; Google retires
# models for new keys over time (gemini-2.5-flash did, mid-project).
GEMINI_MODELS = list(
    dict.fromkeys(
        [
            os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite"),
            "gemini-3.5-flash-lite",
            "gemini-3.8-flash",
        ]
    )
)
GEMINI_MODEL = GEMINI_MODELS[0]
GEMINI_URL_TEMPLATE = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

# Whole chain must finish within this, so a user never waits minutes on a spinner.
TOTAL_BUDGET_SECONDS = 30
PER_REQUEST_TIMEOUT_SECONDS = 15

# Worth trying the next model: overloaded (500/503), per-model rate limit
# (429), or model retired for this key (404).
FALL_THROUGH_STATUS = {404, 429, 500, 503}

# Failures that would hit every model the same way — stop immediately.
FATAL_ERRORS = {
    400: "Gemini rejected the request — usually this means GEMINI_API_KEY is invalid.",
    403: "This API key doesn't have access to the Gemini API — create a new key in AI Studio.",
}


class GeminiUnavailable(RuntimeError):
    """Every model in the chain was busy, rate-limited, retired or too slow."""


SYSTEM_INSTRUCTION = """You are a finance operations analyst assistant.

You will be given a JSON object of ALREADY-COMPUTED KPI figures and anomaly \
records from a company's vendor invoice data. Follow these rules exactly:

1. Only reference numbers and records that appear in the JSON provided. \
Never invent, estimate, or infer a number that is not explicitly present.
2. Never perform arithmetic, aggregation, or any calculation yourself — \
every figure you cite must be copied directly from the JSON.
3. You have not seen and do not have access to any raw transaction-level \
data beyond what is in the JSON below. Do not imply otherwise.
4. If the JSON does not contain enough information to answer the question, \
say so plainly instead of guessing.
5. Write a concise, professional summary (3-6 sentences) for a finance \
operations stakeholder. Plain language, no code or JSON syntax."""


def build_request_body(facts: dict) -> dict:
    user_content = (
        f"Question: {facts.get('question', '(no question — general overview)')}\n\n"
        f"Precomputed facts (JSON):\n{json.dumps(facts, indent=2, default=str)}"
    )
    return {
        "system_instruction": {"parts": [{"text": SYSTEM_INSTRUCTION}]},
        "contents": [{"role": "user", "parts": [{"text": user_content}]}],
        # Newer Gemini models "think" before answering, and those thought
        # tokens count against this limit — at 512, gemini-3.8-flash spent 492
        # on thinking and returned a 16-token, mid-sentence answer.
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 4096},
    }


def generate_insight_with_model(facts: dict, api_key: str = None) -> tuple[str, str]:
    """Narrate the facts with the first model in GEMINI_MODELS that answers.

    Returns (text, model_used). Raises GeminiUnavailable if no model answers
    within the time budget, or RuntimeError for errors no model can fix.
    """
    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and add your "
            "Google AI Studio key (aistudio.google.com)."
        )

    deadline = time.monotonic() + TOTAL_BUDGET_SECONDS
    body = build_request_body(facts)
    for model in GEMINI_MODELS:
        remaining = deadline - time.monotonic()
        if remaining <= 1:
            break
        try:
            response = requests.post(
                GEMINI_URL_TEMPLATE.format(model=model),
                headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
                json=body,
                timeout=min(PER_REQUEST_TIMEOUT_SECONDS, remaining),
            )
        except (requests.Timeout, requests.ConnectionError):
            continue

        if response.status_code in FALL_THROUGH_STATUS:
            continue
        if response.status_code in FATAL_ERRORS:
            raise RuntimeError(FATAL_ERRORS[response.status_code])
        response.raise_for_status()

        payload = response.json()
        candidates = payload.get("candidates")
        if not candidates:
            block_reason = payload.get("promptFeedback", {}).get("blockReason")
            raise RuntimeError(f"Gemini returned no candidates (blockReason={block_reason})")
        if candidates[0].get("finishReason") == "MAX_TOKENS":
            continue  # cut off mid-answer: never show a truncated narrative
        return candidates[0]["content"]["parts"][0]["text"].strip(), model

    raise GeminiUnavailable(
        "Gemini's free tier is under high demand right now — please try again in a minute."
    )


def generate_insight(facts: dict, api_key: str = None) -> str:
    return generate_insight_with_model(facts, api_key)[0]


if __name__ == "__main__":
    from src.retrieval import retrieve

    facts = retrieve("Any budget overruns?")
    print(generate_insight(facts))
