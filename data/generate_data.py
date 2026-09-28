"""Synthetic finance-ops data generator.

Produces vendor invoices, monthly department/category budgets, and a
ground-truth anomaly label file. Ground truth is written separately from
the raw invoice/budget data on purpose: real invoice data would never carry
a "this row is fraudulent" column, and src/anomalies.py must detect these
cases from statistical signal alone, not read the label. Ground truth
exists only so tests/test_kpis.py can check detection results against
what was actually planted.
"""

import random
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

SEED = 42
N_VENDORS = 40
N_BASE_INVOICES = 2500
MONTHS_BACK = 12

DEPARTMENTS = ["Finance", "IT", "Marketing", "Operations", "HR", "Legal"]
CATEGORIES = ["Consulting", "Software & IT Services", "Facilities", "Travel", "Office Supplies", "Professional Services"]
CURRENCY = "EUR"

DUPLICATE_RATE = 0.02
PROCESSING_OUTLIER_RATE = 0.03
OVERRUN_GROUP_RATE = 0.10

DEPT_CASE_MESSY_RATE = 0.15
CATEGORY_CASE_MESSY_RATE = 0.15
STATUS_CASE_MESSY_RATE = 0.10
DATE_FORMAT_MESSY_RATE = 0.08
MISSING_SUBMITTED_DATE_RATE = 0.01
VENDOR_NAME_WHITESPACE_RATE = 0.15

OUT_DIR = Path(__file__).parent / "raw"


def make_vendors(fake: Faker) -> pd.DataFrame:
    rows = []
    for i in range(1, N_VENDORS + 1):
        rows.append({
            "vendor_id": f"V{i:03d}",
            "vendor_name": fake.company(),
            "category": random.choice(CATEGORIES),
        })
    return pd.DataFrame(rows)


def make_budgets(end_date: pd.Timestamp) -> pd.DataFrame:
    months = pd.date_range(end=end_date, periods=MONTHS_BACK, freq="MS")
    rows = []
    for month in months:
        for dept in DEPARTMENTS:
            for cat in CATEGORIES:
                base = random.uniform(8000, 35000)
                rows.append({
                    "department": dept,
                    "category": cat,
                    "month": month.strftime("%Y-%m"),
                    "budget_amount": round(base, 2),
                })
    return pd.DataFrame(rows)


def make_base_invoices(fake: Faker, vendors: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    rows = []
    date_span_days = (end_date - start_date).days

    for i in range(1, N_BASE_INVOICES + 1):
        vendor = vendors.sample(1).iloc[0]
        dept = random.choice(DEPARTMENTS)

        invoice_date = start_date + pd.Timedelta(days=random.randint(0, date_span_days))
        submitted_date = invoice_date + pd.Timedelta(days=random.randint(0, 5))
        approval_lag = max(1, int(np.random.gamma(shape=2.0, scale=2.5)))
        approved_date = submitted_date + pd.Timedelta(days=approval_lag)
        payment_lag = max(1, int(np.random.gamma(shape=2.0, scale=2.0)))
        paid_date = approved_date + pd.Timedelta(days=payment_lag)

        amount = round(float(np.random.lognormal(mean=6.5, sigma=1.0)), 2)
        amount = min(amount, 50000.0)

        rows.append({
            "invoice_id": f"INV{i:05d}",
            "vendor_id": vendor["vendor_id"],
            "department": dept,
            "category": vendor["category"],
            "invoice_date": invoice_date.date().isoformat(),
            "submitted_date": submitted_date.date().isoformat(),
            "approved_date": approved_date.date().isoformat(),
            "paid_date": paid_date.date().isoformat(),
            "amount": amount,
            "currency": CURRENCY,
            "status": "Paid",
        })

    return pd.DataFrame(rows)


def inject_duplicate_payments(invoices: pd.DataFrame, next_id: int) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    n = max(1, int(len(invoices) * DUPLICATE_RATE))
    originals = invoices.sample(n, random_state=SEED)

    dup_rows = []
    ground_truth = []
    for _, orig in originals.iterrows():
        dup = orig.copy()
        dup["invoice_id"] = f"INV{next_id:05d}"
        next_id += 1

        orig_date = pd.Timestamp(orig["invoice_date"])
        dup["invoice_date"] = (orig_date + pd.Timedelta(days=random.randint(0, 2))).date().isoformat()
        dup["submitted_date"] = (pd.Timestamp(dup["invoice_date"]) + pd.Timedelta(days=random.randint(0, 3))).date().isoformat()
        dup["approved_date"] = (pd.Timestamp(dup["submitted_date"]) + pd.Timedelta(days=random.randint(1, 5))).date().isoformat()
        dup["paid_date"] = (pd.Timestamp(dup["approved_date"]) + pd.Timedelta(days=random.randint(1, 4))).date().isoformat()

        dup_rows.append(dup)
        ground_truth.append({
            "anomaly_type": "duplicate_payment",
            "invoice_id": dup["invoice_id"],
            "department": None,
            "category": None,
            "month": None,
            "detail": f"duplicate of {orig['invoice_id']}",
        })

    return pd.DataFrame(dup_rows), pd.DataFrame(ground_truth), next_id


def inject_processing_time_outliers(invoices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = max(1, int(len(invoices) * PROCESSING_OUTLIER_RATE))
    idx = invoices.sample(n, random_state=SEED + 1).index

    ground_truth = []
    for i in idx:
        submitted = pd.Timestamp(invoices.at[i, "submitted_date"])
        stretched_paid = submitted + pd.Timedelta(days=random.randint(45, 90))
        invoices.at[i, "approved_date"] = (submitted + pd.Timedelta(days=random.randint(20, 40))).date().isoformat()
        invoices.at[i, "paid_date"] = stretched_paid.date().isoformat()

        ground_truth.append({
            "anomaly_type": "processing_time_outlier",
            "invoice_id": invoices.at[i, "invoice_id"],
            "department": None,
            "category": None,
            "month": None,
            "detail": "submitted-to-paid stretched to 45-90 days",
        })

    return invoices, pd.DataFrame(ground_truth)


def inject_budget_overruns(invoices: pd.DataFrame, budgets: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    invoices = invoices.copy()
    invoices["month"] = pd.to_datetime(invoices["invoice_date"]).dt.strftime("%Y-%m")

    groups = budgets.sample(frac=1.0, random_state=SEED + 2)
    n_groups = max(1, int(len(budgets) * OVERRUN_GROUP_RATE))
    target_groups = groups.head(n_groups)

    ground_truth = []
    for _, g in target_groups.iterrows():
        mask = (
            (invoices["department"] == g["department"])
            & (invoices["category"] == g["category"])
            & (invoices["month"] == g["month"])
        )
        group_idx = invoices[mask].index
        if len(group_idx) == 0:
            continue

        current_total = invoices.loc[group_idx, "amount"].sum()
        target_total = g["budget_amount"] * random.uniform(1.15, 1.6)
        if current_total <= 0:
            continue

        scale = target_total / current_total
        invoices.loc[group_idx, "amount"] = (invoices.loc[group_idx, "amount"] * scale).round(2)

        ground_truth.append({
            "anomaly_type": "budget_overrun",
            "invoice_id": None,
            "department": g["department"],
            "category": g["category"],
            "month": g["month"],
            "detail": f"actual ~{invoices.loc[group_idx, 'amount'].sum():.2f} vs budget {g['budget_amount']:.2f}",
        })

    invoices = invoices.drop(columns=["month"])
    return invoices, pd.DataFrame(ground_truth)


def messify_case(value: str) -> str:
    variant = random.choice(["upper", "lower", "title", "pad"])
    if variant == "upper":
        return value.upper()
    if variant == "lower":
        return value.lower()
    if variant == "title":
        return value.title()
    return f"  {value}  "


def messify_invoices(invoices: pd.DataFrame, protected_ids: set) -> pd.DataFrame:
    """Add presentation-layer mess (casing, whitespace, date formats, a few
    missing timestamps) on top of already-final anomaly rows. Run only after
    duplicate/outlier/overrun injection so the planted anomalies stay intact
    and detectable in amount/date fields; this only touches text formatting.
    """
    invoices = invoices.copy()

    dept_mask = np.random.rand(len(invoices)) < DEPT_CASE_MESSY_RATE
    invoices.loc[dept_mask, "department"] = invoices.loc[dept_mask, "department"].map(messify_case)

    cat_mask = np.random.rand(len(invoices)) < CATEGORY_CASE_MESSY_RATE
    invoices.loc[cat_mask, "category"] = invoices.loc[cat_mask, "category"].map(messify_case)

    status_mask = np.random.rand(len(invoices)) < STATUS_CASE_MESSY_RATE
    invoices.loc[status_mask, "status"] = invoices.loc[status_mask, "status"].map(messify_case)

    date_formats = ["%m/%d/%Y", "%d-%b-%Y", "%Y/%m/%d"]
    date_mask = np.random.rand(len(invoices)) < DATE_FORMAT_MESSY_RATE
    for i in invoices[date_mask].index:
        fmt = random.choice(date_formats)
        invoices.at[i, "invoice_date"] = pd.Timestamp(invoices.at[i, "invoice_date"]).strftime(fmt)

    eligible = invoices.index[~invoices["invoice_id"].isin(protected_ids)]
    n_missing = max(1, int(len(eligible) * MISSING_SUBMITTED_DATE_RATE))
    missing_idx = np.random.choice(eligible, size=n_missing, replace=False)
    invoices.loc[missing_idx, "submitted_date"] = None

    return invoices


def messify_vendors(vendors: pd.DataFrame) -> pd.DataFrame:
    def messy_name(name: str) -> str:
        variant = random.choice(["pad", "double_space", "as_is"])
        if variant == "pad":
            return f"  {name}  "
        if variant == "double_space":
            return name.replace(" ", "  ", 1)
        return name

    vendors = vendors.copy()
    mask = np.random.rand(len(vendors)) < VENDOR_NAME_WHITESPACE_RATE
    vendors.loc[mask, "vendor_name"] = vendors.loc[mask, "vendor_name"].map(messy_name)
    return vendors


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    fake = Faker()
    Faker.seed(SEED)

    # Fixed, not today(): with a moving end date, re-running this script
    # shifts every date and anomaly, so data/raw would no longer match the
    # committed data/processed parquet and the ground-truth tests would fail.
    end_date = pd.Timestamp("2026-08-31")
    start_date = end_date - pd.DateOffset(months=MONTHS_BACK)

    vendors = make_vendors(fake)
    budgets = make_budgets(end_date)
    invoices = make_base_invoices(fake, vendors, start_date, end_date)

    next_id = N_BASE_INVOICES + 1
    dup_invoices, dup_truth, next_id = inject_duplicate_payments(invoices, next_id)
    invoices = pd.concat([invoices, dup_invoices], ignore_index=True)

    invoices, outlier_truth = inject_processing_time_outliers(invoices)
    invoices, overrun_truth = inject_budget_overruns(invoices, budgets)

    ground_truth = pd.concat([dup_truth, outlier_truth, overrun_truth], ignore_index=True)

    protected_ids = set(dup_truth["invoice_id"]) | set(outlier_truth["invoice_id"])
    invoices = messify_invoices(invoices, protected_ids)
    vendors = messify_vendors(vendors)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    vendors.to_csv(OUT_DIR / "vendors.csv", index=False)
    budgets.to_csv(OUT_DIR / "budgets.csv", index=False)
    invoices.to_csv(OUT_DIR / "invoices.csv", index=False)
    ground_truth.to_csv(OUT_DIR / "anomaly_ground_truth.csv", index=False)

    print(f"vendors:        {len(vendors)}")
    print(f"budgets:        {len(budgets)}")
    print(f"invoices:       {len(invoices)} (base {N_BASE_INVOICES} + {len(dup_invoices)} duplicates)")
    print(f"ground truth:   {len(ground_truth)} planted anomalies")
    print(f"  duplicate_payment:       {len(dup_truth)}")
    print(f"  processing_time_outlier: {len(outlier_truth)}")
    print(f"  budget_overrun:          {len(overrun_truth)}")
    print(f"written to {OUT_DIR}")


if __name__ == "__main__":
    main()
