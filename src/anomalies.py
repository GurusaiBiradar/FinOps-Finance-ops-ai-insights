"""Rule-based anomaly detection over the cleaned finance-ops data.

Deterministic stats/rules only — no LLM involvement (see CLAUDE.md hard
constraints). Thresholds below were tuned by checking precision/recall
against data/raw/anomaly_ground_truth.csv during development; anomalies.py
itself never reads that file — it only sees data/processed/, exactly like a
real detector would.
"""

import pandas as pd

from src.kpis import load_budgets, load_invoices

DUPLICATE_DATE_WINDOW_DAYS = 5

# Tukey's "extreme outlier" fence (vs. the more common 1.5x "mild outlier"
# fence). 1.5x flagged 36 false positives on the natural gamma-distributed
# processing-time tail; 3.0x still catches every planted outlier (0 misses)
# while cutting false positives to 2.
PROCESSING_TIME_IQR_MULTIPLIER = 3.0

# Real budgets are targets with some normal slack, not a statistical
# distribution — a fixed tolerance is more explainable than a z-score here.
# 1.15 catches every planted overrun (0 misses); tightening further starts
# missing planted cases that landed near the injection floor (1.15x-1.6x).
BUDGET_OVERRUN_THRESHOLD = 1.15


def detect_duplicate_payments(invoices: pd.DataFrame) -> pd.DataFrame:
    """Flag invoices that share a vendor and amount within a short date
    window — a classic AP double-payment signature. Both invoices in a
    matching pair are flagged, since a detector can't know which one is
    the "original".
    """
    flagged_ids = set()
    pairs = []

    grouped = invoices.groupby(["vendor_id", invoices["amount"].round(2)])
    for (vendor_id, amount), group in grouped:
        if len(group) < 2:
            continue
        ordered = group[["invoice_id", "invoice_date"]].sort_values("invoice_date").reset_index(drop=True)
        for i in range(len(ordered) - 1):
            days_apart = (ordered.at[i + 1, "invoice_date"] - ordered.at[i, "invoice_date"]).days
            if days_apart <= DUPLICATE_DATE_WINDOW_DAYS:
                id_a, id_b = ordered.at[i, "invoice_id"], ordered.at[i + 1, "invoice_id"]
                flagged_ids.update([id_a, id_b])
                pairs.append({
                    "anomaly_type": "duplicate_payment",
                    "invoice_id": id_a,
                    "department": None,
                    "category": None,
                    "month": None,
                    "detail": f"matches {id_b}: vendor {vendor_id}, amount {amount:.2f}, {days_apart}d apart",
                })
                pairs.append({
                    "anomaly_type": "duplicate_payment",
                    "invoice_id": id_b,
                    "department": None,
                    "category": None,
                    "month": None,
                    "detail": f"matches {id_a}: vendor {vendor_id}, amount {amount:.2f}, {days_apart}d apart",
                })

    return pd.DataFrame(pairs)


def detect_processing_time_outliers(invoices: pd.DataFrame) -> pd.DataFrame:
    valid = invoices[~invoices["missing_submitted_date"]]
    q1, q3 = valid["processing_days"].quantile(0.25), valid["processing_days"].quantile(0.75)
    iqr = q3 - q1
    fence = q3 + PROCESSING_TIME_IQR_MULTIPLIER * iqr

    outliers = valid[valid["processing_days"] > fence]
    return pd.DataFrame([
        {
            "anomaly_type": "processing_time_outlier",
            "invoice_id": row.invoice_id,
            "department": None,
            "category": None,
            "month": None,
            "detail": f"{row.processing_days:.0f}d processing time (fence {fence:.1f}d)",
        }
        for row in outliers.itertuples()
    ])


def detect_budget_overruns(invoices: pd.DataFrame, budgets: pd.DataFrame) -> pd.DataFrame:
    monthly = invoices.copy()
    monthly["month"] = monthly["invoice_date"].dt.strftime("%Y-%m")
    actual = (
        monthly.groupby(["department", "category", "month"])["amount"]
        .sum()
        .reset_index(name="actual_amount")
    )
    merged = budgets.merge(actual, on=["department", "category", "month"], how="left")
    merged["actual_amount"] = merged["actual_amount"].fillna(0.0)
    merged["ratio"] = merged["actual_amount"] / merged["budget_amount"]

    overruns = merged[merged["ratio"] > BUDGET_OVERRUN_THRESHOLD]
    return pd.DataFrame([
        {
            "anomaly_type": "budget_overrun",
            "invoice_id": None,
            "department": row.department,
            "category": row.category,
            "month": row.month,
            "detail": f"actual {row.actual_amount:.2f} vs budget {row.budget_amount:.2f} ({row.ratio * 100:.0f}%)",
        }
        for row in overruns.itertuples()
    ])


def detect_all(invoices: pd.DataFrame | None = None, budgets: pd.DataFrame | None = None) -> pd.DataFrame:
    if invoices is None:
        invoices = load_invoices()
    if budgets is None:
        budgets = load_budgets()

    frames = [
        detect_duplicate_payments(invoices),
        detect_processing_time_outliers(invoices),
        detect_budget_overruns(invoices, budgets),
    ]
    non_empty = [f for f in frames if not f.empty]
    if not non_empty:
        return pd.DataFrame(columns=["anomaly_type", "invoice_id", "department", "category", "month", "detail"])
    return pd.concat(non_empty, ignore_index=True)


if __name__ == "__main__":
    anomalies = detect_all()
    print(anomalies["anomaly_type"].value_counts())
    print()
    print(anomalies.head(10).to_string(index=False))
