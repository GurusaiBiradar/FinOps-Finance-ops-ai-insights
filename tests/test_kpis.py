from pathlib import Path

import pandas as pd
import pytest

from src import anomalies, kpis

GROUND_TRUTH_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "anomaly_ground_truth.csv"


@pytest.fixture(scope="module")
def invoices():
    return kpis.load_invoices()


@pytest.fixture(scope="module")
def budgets():
    return kpis.load_budgets()


@pytest.fixture(scope="module")
def ground_truth():
    return pd.read_csv(GROUND_TRUTH_PATH)


@pytest.fixture(scope="module")
def detected(invoices, budgets):
    return anomalies.detect_all(invoices, budgets)


# --- KPI sanity checks ------------------------------------------------

def test_total_spend_matches_sum_of_amounts(invoices):
    assert kpis.total_spend(invoices) == pytest.approx(invoices["amount"].sum(), abs=0.01)


def test_spend_by_department_covers_every_invoice(invoices):
    by_dept = kpis.spend_by_department(invoices)
    assert by_dept["invoice_count"].sum() == len(invoices)
    assert by_dept["total_spend"].sum() == pytest.approx(kpis.total_spend(invoices), abs=0.5)


def test_avg_processing_days_excludes_missing_submitted_date(invoices):
    result = kpis.avg_processing_days(invoices)
    valid = invoices[~invoices["missing_submitted_date"]]
    assert result == pytest.approx(valid["processing_days"].mean(), abs=0.1)


def test_budget_vs_actual_covers_every_budget_row(invoices, budgets):
    result = kpis.budget_vs_actual(invoices, budgets)
    assert len(result) == len(budgets)


# --- Anomaly detection vs. known planted cases -------------------------
# Ground truth is only used here, in tests, to check what anomalies.py
# found against what generate_data.py actually planted. anomalies.py never
# reads this file itself.

def test_duplicate_payment_detection_matches_ground_truth_exactly(detected, ground_truth):
    dup_truth = ground_truth[ground_truth["anomaly_type"] == "duplicate_payment"]
    planted_dup_ids = set(dup_truth["invoice_id"])
    planted_original_ids = set(dup_truth["detail"].str.split().str[-1])
    expected_pair_members = planted_dup_ids | planted_original_ids

    detected_ids = set(detected.loc[detected["anomaly_type"] == "duplicate_payment", "invoice_id"])
    assert detected_ids == expected_pair_members


def test_processing_time_outlier_detection_has_no_false_negatives(detected, ground_truth):
    planted = set(ground_truth.loc[ground_truth["anomaly_type"] == "processing_time_outlier", "invoice_id"])
    detected_ids = set(detected.loc[detected["anomaly_type"] == "processing_time_outlier", "invoice_id"])
    assert planted <= detected_ids


def test_processing_time_outlier_detection_precision_is_reasonable(detected, ground_truth):
    planted = set(ground_truth.loc[ground_truth["anomaly_type"] == "processing_time_outlier", "invoice_id"])
    detected_ids = set(detected.loc[detected["anomaly_type"] == "processing_time_outlier", "invoice_id"])
    false_positives = detected_ids - planted
    assert len(false_positives) <= 10


def test_budget_overrun_detection_has_no_false_negatives(detected, ground_truth):
    overrun_truth = ground_truth[ground_truth["anomaly_type"] == "budget_overrun"]
    planted = set(zip(overrun_truth["department"], overrun_truth["category"], overrun_truth["month"]))

    overrun_detected = detected[detected["anomaly_type"] == "budget_overrun"]
    detected_keys = set(zip(overrun_detected["department"], overrun_detected["category"], overrun_detected["month"]))
    assert planted <= detected_keys


def test_budget_overrun_detection_precision_is_reasonable(detected, ground_truth):
    overrun_truth = ground_truth[ground_truth["anomaly_type"] == "budget_overrun"]
    planted = set(zip(overrun_truth["department"], overrun_truth["category"], overrun_truth["month"]))

    overrun_detected = detected[detected["anomaly_type"] == "budget_overrun"]
    detected_keys = set(zip(overrun_detected["department"], overrun_detected["category"], overrun_detected["month"]))
    false_positives = detected_keys - planted
    assert len(false_positives) <= 15
