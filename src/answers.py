"""Answer orchestration: live Gemini first, pre-generated answer as last resort.

Order of preference for any question:
  1. Live — src/insights.py tries each model in its fallback chain.
  2. Pre-generated — only if every live model failed, and only for the
     briefing + suggested questions, whose answers were generated earlier by
     scripts/pregenerate_answers.py and saved to data/pregenerated/answers.json.

A saved answer is keyed by a fingerprint of the exact facts it narrated, so if
the data or retrieval logic changes, stale answers are simply never shown.
Every result carries its source, so the UI can label it honestly.
"""

import hashlib
import json
from pathlib import Path

from src.insights import GeminiUnavailable, generate_insight_with_model

PREGENERATED_PATH = Path(__file__).resolve().parent.parent / "data" / "pregenerated" / "answers.json"

BRIEFING_QUESTION = (
    "Give a short company-wide briefing: overall spend, the biggest budget overruns, "
    "duplicate payments, and processing delays."
)
SUGGESTED_QUESTIONS = [
    "Any budget overruns?",
    "What duplicate payments were found?",
    "How much did we spend on Consulting?",
    "Where are invoices getting delayed?",
]
PREGENERATED_QUESTIONS = [BRIEFING_QUESTION, *SUGGESTED_QUESTIONS]


def facts_fingerprint(facts: dict) -> str:
    canonical = json.dumps(facts, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_pregenerated() -> dict:
    if not PREGENERATED_PATH.exists():
        return {}
    return json.loads(PREGENERATED_PATH.read_text(encoding="utf-8"))


def find_pregenerated(facts: dict) -> dict | None:
    return load_pregenerated().get(facts_fingerprint(facts))


def answer(facts: dict) -> dict:
    """Returns {"text", "source": "live" | "pregenerated", "model", "generated_on"}."""
    try:
        text, model = generate_insight_with_model(facts)
        return {"text": text, "source": "live", "model": model, "generated_on": None}
    except GeminiUnavailable:
        saved = find_pregenerated(facts)
        if saved is None:
            raise
        return {"text": saved["text"], "source": "pregenerated", "model": saved["model"], "generated_on": saved["generated_on"]}
