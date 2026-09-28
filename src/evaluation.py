"""Scores the rule-based detectors against the generator's answer key.

Evaluation only: src/anomalies.py never reads the ground-truth file. This
module exists so the app can show how well detection actually performs, with
numbers computed live rather than copied into the UI by hand.
"""

from pathlib import Path

import pandas as pd

GROUND_TRUTH_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "anomaly_ground_truth.csv"
ANOMALY_TYPES = ["duplicate_payment", "processing_time_outlier", "budget_overrun"]


def load_ground_truth() -> pd.DataFrame:
    return pd.read_csv(GROUND_TRUTH_PATH)


def _keys(df: pd.DataFrame, anomaly_type: str) -> set:
    rows = df[df["anomaly_type"] == anomaly_type]
    if anomaly_type == "budget_overrun":
        # Overruns are department/category/month groups, not single invoices.
        return set(zip(rows["department"], rows["category"], rows["month"]))
    return set(rows["invoice_id"])


def detection_scorecard(detected: pd.DataFrame, ground_truth: pd.DataFrame) -> pd.DataFrame:
    records = []
    for anomaly_type in ANOMALY_TYPES:
        planted = _keys(ground_truth, anomaly_type)
        if anomaly_type == "duplicate_payment":
            # The answer key lists each planted copy ("duplicate of INV01448");
            # the detector flags both halves of a pair, so originals count too.
            copies = ground_truth[ground_truth["anomaly_type"] == anomaly_type]
            planted |= set(copies["detail"].str.split().str[-1])

        flagged = _keys(detected, anomaly_type)
        caught = planted & flagged
        records.append(
            {
                "anomaly_type": anomaly_type,
                "planted": len(planted),
                "flagged": len(flagged),
                "caught": len(caught),
                "false_positives": len(flagged - planted),
                "recall": len(caught) / len(planted) if planted else 0.0,
                "precision": len(caught) / len(flagged) if flagged else 0.0,
            }
        )
    return pd.DataFrame(records)
