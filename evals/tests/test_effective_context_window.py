from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from effective_context_window import BINS, assign_bin, compute_degradation_curve


def test_assign_bin_first_bucket():
    assert assign_bin(1_000, BINS) == "<5K"


def test_assign_bin_boundary_at_5k_is_second_bucket():
    assert assign_bin(5_000, BINS) == "5K-12K"


def test_assign_bin_just_before_12k():
    assert assign_bin(11_999, BINS) == "5K-12K"


def test_assign_bin_at_12k_is_third_bucket():
    assert assign_bin(12_000, BINS) == "12K-30K"


def test_assign_bin_last_bucket():
    assert assign_bin(99_000, BINS) == "60K-100K"


def test_assign_bin_out_of_range_returns_none():
    assert assign_bin(200_000, BINS) is None


def test_assign_bin_zero():
    assert assign_bin(0, BINS) == "<5K"


def test_compute_curve_means():
    rows = [
        {"token_count": 1_000, "score": 0.9},
        {"token_count": 2_000, "score": 0.8},
        {"token_count": 8_000, "score": 0.7},
        {"token_count": 20_000, "score": 0.5},
    ]
    result = compute_degradation_curve(rows, BINS)
    assert result["<5K"]["mean"] == pytest.approx(0.85)
    assert result["5K-12K"]["mean"] == pytest.approx(0.7)
    assert result["12K-30K"]["mean"] == pytest.approx(0.5)
    assert result["30K-60K"]["n"] == 0
    assert result["30K-60K"]["mean"] is None


def test_compute_curve_counts():
    rows = [
        {"token_count": 1_000, "score": 0.9},
        {"token_count": 1_500, "score": 0.8},
        {"token_count": 6_000, "score": 0.7},
    ]
    result = compute_degradation_curve(rows, BINS)
    assert result["<5K"]["n"] == 2
    assert result["5K-12K"]["n"] == 1
    assert result["12K-30K"]["n"] == 0


def test_compute_curve_empty_rows():
    result = compute_degradation_curve([], BINS)
    for _, _, label in BINS:
        assert result[label]["n"] == 0
        assert result[label]["mean"] is None


def test_compute_curve_out_of_range_docs_ignored():
    rows = [{"token_count": 200_000, "score": 0.5}]
    result = compute_degradation_curve(rows, BINS)
    assert all(b["n"] == 0 for b in result.values())
