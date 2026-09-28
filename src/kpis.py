"""Deterministic KPI calculations over the cleaned finance-ops data.

Pure aggregation only — no anomaly logic (see anomalies.py) and no LLM calls.
Everything here must be independently reproducible from data/processed/.
"""

from pathlib import Path

import pandas as pd

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


def load_invoices() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / "invoices_clean.parquet")


def load_budgets() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / "budgets.parquet")


def total_spend(invoices: pd.DataFrame) -> float:
    return round(float(invoices["amount"].sum()), 2)


def spend_by_department(invoices: pd.DataFrame) -> pd.DataFrame:
    return (
        invoices.groupby("department")
        .agg(total_spend=("amount", "sum"), invoice_count=("invoice_id", "count"))
        .round(2)
        .reset_index()
        .sort_values("total_spend", ascending=False)
    )


def spend_by_category(invoices: pd.DataFrame) -> pd.DataFrame:
    return (
        invoices.groupby("category")
        .agg(total_spend=("amount", "sum"), invoice_count=("invoice_id", "count"))
        .round(2)
        .reset_index()
        .sort_values("total_spend", ascending=False)
    )


def spend_by_vendor(invoices: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    result = (
        invoices.groupby(["vendor_id", "vendor_name"])
        .agg(total_spend=("amount", "sum"), invoice_count=("invoice_id", "count"))
        .round(2)
        .reset_index()
        .sort_values("total_spend", ascending=False)
    )
    return result.head(top_n)


def spend_by_month(invoices: pd.DataFrame) -> pd.DataFrame:
    monthly = invoices.copy()
    monthly["month"] = monthly["invoice_date"].dt.strftime("%Y-%m")
    return (
        monthly.groupby("month")
        .agg(total_spend=("amount", "sum"), invoice_count=("invoice_id", "count"))
        .round(2)
        .reset_index()
        .sort_values("month")
    )


def avg_processing_days(invoices: pd.DataFrame, by: str | None = None) -> pd.DataFrame | float:
    valid = invoices[~invoices["missing_submitted_date"]]
    if by is None:
        return round(float(valid["processing_days"].mean()), 1)
    return (
        valid.groupby(by)["processing_days"]
        .mean()
        .round(1)
        .reset_index(name="avg_processing_days")
        .sort_values("avg_processing_days", ascending=False)
    )


def budget_vs_actual(invoices: pd.DataFrame, budgets: pd.DataFrame) -> pd.DataFrame:
    monthly = invoices.copy()
    monthly["month"] = monthly["invoice_date"].dt.strftime("%Y-%m")
    actual = (
        monthly.groupby(["department", "category", "month"])["amount"]
        .sum()
        .reset_index(name="actual_amount")
    )
    merged = budgets.merge(actual, on=["department", "category", "month"], how="left")
    merged["actual_amount"] = merged["actual_amount"].fillna(0.0).round(2)
    merged["variance"] = (merged["actual_amount"] - merged["budget_amount"]).round(2)
    # Round after scaling: rounding first then * 100 reintroduces float noise
    # (59.160000000000004) that ends up verbatim in the facts sent to Gemini.
    merged["variance_pct"] = (((merged["actual_amount"] / merged["budget_amount"]) - 1) * 100).round(2)
    return merged.sort_values("variance_pct", ascending=False)


def budget_variance_extremes(invoices: pd.DataFrame, budgets: pd.DataFrame, n: int = 6) -> pd.DataFrame:
    """The n most-over-budget and n most-under-budget department/category/month
    groups. A full-department annual aggregate washes out to "under budget"
    almost everywhere, since overruns are localized to specific months/categories
    — this instead surfaces the actual extremes the anomaly detector flags on.
    """
    bva = budget_vs_actual(invoices, budgets)
    extremes = pd.concat([bva.head(n), bva.tail(n)]).drop_duplicates()
    extremes = extremes.copy()
    extremes["label"] = extremes["department"] + " · " + extremes["category"] + " (" + extremes["month"] + ")"
    return extremes.sort_values("variance_pct", ascending=False)


def compute_all_kpis(invoices: pd.DataFrame | None = None, budgets: pd.DataFrame | None = None) -> dict:
    if invoices is None:
        invoices = load_invoices()
    if budgets is None:
        budgets = load_budgets()

    return {
        "total_spend": total_spend(invoices),
        "spend_by_department": spend_by_department(invoices),
        "spend_by_category": spend_by_category(invoices),
        "spend_by_vendor": spend_by_vendor(invoices),
        "spend_by_month": spend_by_month(invoices),
        "avg_processing_days": avg_processing_days(invoices),
        "avg_processing_days_by_department": avg_processing_days(invoices, by="department"),
        "budget_vs_actual": budget_vs_actual(invoices, budgets),
    }


if __name__ == "__main__":
    kpis = compute_all_kpis()
    print(f"total_spend: EUR {kpis['total_spend']:,.2f}")
    print(f"avg_processing_days: {kpis['avg_processing_days']}")
    print()
    print(kpis["spend_by_department"].to_string(index=False))
