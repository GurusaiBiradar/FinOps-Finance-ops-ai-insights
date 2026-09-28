"""RAG-lite retrieval layer.

Real RAG: embed a question, run vector similarity search over a document
store, hand the top-k chunks to an LLM. There's no vector DB here on
purpose — the "documents" are a handful of already-computed KPI tables and
anomaly records (data/processed/), not unstructured text, and the topics a
user can ask about are a small enumerable set. So retrieval here is
keyword/topic matching over those precomputed facts, doing the same job
(narrow the context before it reaches an LLM) without the machinery.

This module never calls Gemini and never touches raw invoice rows — it only
narrows down which *already-computed* facts are relevant. See src/insights.py
for the LLM call that narrates whatever this returns.
"""

import pandas as pd

from src.anomalies import detect_all
from src.kpis import (
    avg_processing_days,
    budget_vs_actual,
    load_budgets,
    load_invoices,
    spend_by_category,
    spend_by_department,
    spend_by_vendor,
    total_spend,
)

TOPIC_KEYWORDS = {
    "duplicate_payment": ["duplicate", "double pay", "double-pay", "paid twice"],
    "processing_time_outlier": ["processing time", "delay", "slow", "late", "stuck", "bottleneck", "turnaround"],
    "budget_overrun": ["budget", "overrun", "over budget", "overspend", "variance"],
    "vendor": ["vendor", "supplier"],
    "spend": ["spend", "spending", "cost", "expense", "how much"],
}
ANOMALY_TOPICS = {"duplicate_payment", "processing_time_outlier", "budget_overrun"}
MAX_ROWS = 15


def _detect_topics(question: str) -> set:
    q = question.lower()
    matched = {topic for topic, keywords in TOPIC_KEYWORDS.items() if any(kw in q for kw in keywords)}
    return matched or {"overview"}


def _detect_entity(question: str, values) -> str | None:
    q = question.lower()
    for value in sorted(set(values), key=len, reverse=True):
        if value.lower() in q:
            return value
    return None


def enrich_anomalies_with_dims(anomalies_df: pd.DataFrame, invoices: pd.DataFrame) -> pd.DataFrame:
    lookup = invoices.set_index("invoice_id")[["department", "category"]]
    enriched = anomalies_df.copy()
    has_invoice = enriched["invoice_id"].notna()
    enriched.loc[has_invoice, ["department", "category"]] = lookup.reindex(
        enriched.loc[has_invoice, "invoice_id"]
    ).values
    return enriched


def retrieve(question: str, invoices: pd.DataFrame = None, budgets: pd.DataFrame = None) -> dict:
    """Return only the precomputed facts relevant to a free-text question."""
    if invoices is None:
        invoices = load_invoices()
    if budgets is None:
        budgets = load_budgets()

    topics = _detect_topics(question)
    department = _detect_entity(question, invoices["department"])
    category = _detect_entity(question, invoices["category"])

    scoped = invoices
    if department:
        scoped = scoped[scoped["department"] == department]
    if category:
        scoped = scoped[scoped["category"] == category]

    facts = {
        "question": question,
        "topics": sorted(topics),
        "filters": {"department": department, "category": category},
        "total_spend": total_spend(scoped),
        "avg_processing_days": avg_processing_days(scoped),
    }

    if "spend" in topics or "overview" in topics:
        facts["spend_by_department"] = spend_by_department(scoped).to_dict("records")
        facts["spend_by_category"] = spend_by_category(scoped).to_dict("records")

    if "vendor" in topics:
        facts["top_vendors"] = spend_by_vendor(scoped).to_dict("records")

    if "budget_overrun" in topics or "overview" in topics:
        bva = budget_vs_actual(invoices, budgets)
        if department:
            bva = bva[bva["department"] == department]
        if category:
            bva = bva[bva["category"] == category]
        facts["budget_vs_actual"] = bva.sort_values("variance_pct", ascending=False).head(MAX_ROWS).to_dict("records")

    anomalies_df = enrich_anomalies_with_dims(detect_all(invoices, budgets), invoices)
    if department:
        anomalies_df = anomalies_df[anomalies_df["department"] == department]
    if category:
        anomalies_df = anomalies_df[anomalies_df["category"] == category]

    anomaly_types_wanted = topics & ANOMALY_TOPICS
    if anomaly_types_wanted:
        anomalies_df = anomalies_df[anomalies_df["anomaly_type"].isin(anomaly_types_wanted)]

    facts["anomaly_counts"] = anomalies_df["anomaly_type"].value_counts().to_dict()
    facts["anomalies"] = anomalies_df.head(MAX_ROWS).to_dict("records")

    return facts


if __name__ == "__main__":
    import json

    for q in [
        "What duplicate payments were found in Marketing?",
        "How much did we spend on Consulting?",
        "Any budget overruns?",
        "What's the average processing time?",
    ]:
        print(f"Q: {q}")
        result = retrieve(q)
        print(json.dumps({k: v for k, v in result.items() if k != "anomalies"}, indent=2, default=str))
        print(f"anomalies returned: {len(result['anomalies'])}")
        print()
