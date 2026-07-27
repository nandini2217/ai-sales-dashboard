"""
Unit tests for scripts/metrics.py.

Run from the repo root with: pytest tests/
"""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from metrics import compute_metrics  # noqa: E402


@pytest.fixture
def sample_df():
    """A small, hand-built dataset with a known answer for every metric,
    so each assertion below is checking an exact expected value rather
    than just 'it didn't crash'."""
    return pd.DataFrame(
        {
            "Order ID": ["O1", "O1", "O2", "O3", "O4"],
            "Customer Name": ["Alice", "Alice", "Bob", "Carol", "Dave"],
            "Region": ["West", "West", "East", "West", "South"],
            "Category": ["Technology", "Furniture", "Furniture", "Technology", "Office Supplies"],
            "Sub-Category": ["Phones", "Tables", "Tables", "Phones", "Paper"],
            "Sales": [1000.0, 200.0, 150.0, 500.0, 50.0],
            "Profit": [300.0, -50.0, -40.0, 100.0, 10.0],
            "Discount": [0.0, 0.5, 0.4, 0.1, 0.0],
            "Order Year": [2022, 2022, 2022, 2023, 2023],
            "Order Month": [1, 1, 6, 3, 3],
        }
    )


def test_total_sales_and_profit(sample_df):
    m = compute_metrics(sample_df)
    assert m["total_sales"] == pytest.approx(1900.0)
    assert m["total_profit"] == pytest.approx(320.0)


def test_profit_margin(sample_df):
    m = compute_metrics(sample_df)
    assert m["profit_margin"] == pytest.approx(320.0 / 1900.0 * 100)


def test_order_and_customer_counts(sample_df):
    m = compute_metrics(sample_df)
    # O1 appears twice but is one order; 4 distinct customers
    assert m["total_orders"] == 4
    assert m["total_customers"] == 4


def test_top_region(sample_df):
    m = compute_metrics(sample_df)
    # West = 1000 + 200 + 500 = 1700, highest of any region
    assert m["top_region"] == "West"
    assert m["top_region_sales"] == pytest.approx(1700.0)


def test_loss_making_subcategories(sample_df):
    m = compute_metrics(sample_df)
    # Only "Tables" has negative total profit (-50 + -40 = -90)
    assert m["loss_list"] == "Tables"


def test_best_and_worst_category(sample_df):
    m = compute_metrics(sample_df)
    # Technology: 300 + 100 = 400 (best). Furniture: -50 + -40 = -90 (worst)
    assert m["best_category"] == "Technology"
    assert m["worst_category"] == "Furniture"


def test_best_period(sample_df):
    m = compute_metrics(sample_df)
    # (2022, 1) has 1000 + 200 = 1200 in sales, the highest of any period
    assert m["best_period"] == (2022, 1)


def test_no_loss_making_subcategories_returns_none():
    df = pd.DataFrame(
        {
            "Order ID": ["O1"],
            "Customer Name": ["Alice"],
            "Region": ["West"],
            "Category": ["Technology"],
            "Sub-Category": ["Phones"],
            "Sales": [100.0],
            "Profit": [20.0],
            "Discount": [0.0],
            "Order Year": [2023],
            "Order Month": [1],
        }
    )
    m = compute_metrics(df)
    assert m["loss_list"] == "None"


def test_discount_margin_correlation_is_negative(sample_df):
    # Heavier discounts should coincide with lower margins in this sample
    m = compute_metrics(sample_df)
    assert m["discount_margin_corr"] < 0
