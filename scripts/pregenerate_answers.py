"""Generate the fallback answers the app shows when Gemini is unavailable.

Run whenever the data or retrieval logic changes (the app ignores saved
answers whose facts no longer match):

    python scripts/pregenerate_answers.py           # fill in missing answers
    python scripts/pregenerate_answers.py --force   # regenerate all

Patient by design: the free tier is often busy, so each question is retried
until some model in the chain answers. Progress is saved after every answer.
"""

import json
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.answers import PREGENERATED_PATH, PREGENERATED_QUESTIONS, facts_fingerprint, load_pregenerated  # noqa: E402
from src.insights import GeminiUnavailable, generate_insight_with_model  # noqa: E402
from src.retrieval import retrieve  # noqa: E402

MAX_TRIES_PER_QUESTION = 40
WAIT_BETWEEN_TRIES_SECONDS = 20


def save(entries: dict) -> None:
    PREGENERATED_PATH.parent.mkdir(parents=True, exist_ok=True)
    PREGENERATED_PATH.write_text(json.dumps(entries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main(force: bool) -> None:
    existing = load_pregenerated()
    facts_by_question = {q: retrieve(q) for q in PREGENERATED_QUESTIONS}
    current = {facts_fingerprint(f): q for q, f in facts_by_question.items()}

    # Keep only answers that still match today's facts.
    entries = {fp: e for fp, e in existing.items() if fp in current and not force}
    save(entries)

    for question, facts in facts_by_question.items():
        fingerprint = facts_fingerprint(facts)
        if fingerprint in entries:
            print(f"up to date: {question}")
            continue
        for attempt in range(1, MAX_TRIES_PER_QUESTION + 1):
            try:
                text, model = generate_insight_with_model(facts)
            except GeminiUnavailable:
                print(f"  busy (try {attempt}/{MAX_TRIES_PER_QUESTION}), waiting {WAIT_BETWEEN_TRIES_SECONDS}s...")
                time.sleep(WAIT_BETWEEN_TRIES_SECONDS)
                continue
            entries[fingerprint] = {
                "question": question,
                "text": text,
                "model": model,
                "generated_on": date.today().isoformat(),
            }
            save(entries)
            print(f"saved ({model}): {question}")
            break
        else:
            print(f"GAVE UP: {question}")

    print(f"{len(entries)}/{len(PREGENERATED_QUESTIONS)} answers saved to {PREGENERATED_PATH}")


if __name__ == "__main__":
    main(force="--force" in sys.argv)
