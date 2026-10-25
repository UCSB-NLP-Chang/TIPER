import json
import math

import pytest

import cresta
import cresta.evaluation.usi as usi_api
from cresta.evaluation.usi.sim2real_full_table import (
    DEFAULT_HUMAN_FILE,
    load_human,
)


def test_usi_uses_the_six_term_paper_formula():
    score = usi_api.usi(87.4, 97.9, 88.0, 93.5, 97.4, 0.069)
    expected = (87.4 + 97.9 + 88.0 + 93.5 + 97.4 + 93.1) / 6
    assert score == pytest.approx(expected)
    assert round(score, 1) == 92.9


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        (0.0, 0.0, 100.0),
        (0.0, 5.0, 0.0),
        (5.0, 0.0, 0.0),
        (10.0, 10.0, 100.0),
        (2.0, 6.0, 50.0),
    ],
)
def test_dice_edge_cases(left, right, expected):
    assert usi_api.dice(left, right) == pytest.approx(expected)
    assert usi_api.dice(right, left) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("left", "right"),
    [(-1.0, 1.0), (1.0, -1.0), (math.nan, 1.0), (math.inf, 1.0)],
)
def test_dice_rejects_invalid_inputs(left, right):
    with pytest.raises(ValueError):
        usi_api.dice(left, right)


def test_atomic_metrics_are_available_from_package_api():
    metrics = usi_api.metrics_for_interaction(["Please help", "okay"])
    assert metrics["Wds/trn"] == pytest.approx(1.5)
    assert metrics["Short%"] == pytest.approx(100.0)
    assert metrics["Polite%"] == pytest.approx(50.0)
    assert metrics["Ack%"] == pytest.approx(50.0)


def test_packages_and_public_api_import():
    assert cresta.__doc__
    assert usi_api.__all__ == ["dice", "metrics_for_interaction", "usi"]
    assert callable(usi_api.usi)
    assert callable(usi_api.dice)
    assert callable(usi_api.metrics_for_interaction)


def test_default_human_data_exists_and_loads():
    assert DEFAULT_HUMAN_FILE.is_file()
    with DEFAULT_HUMAN_FILE.open(encoding="utf-8") as f:
        raw = json.load(f)
    assert isinstance(raw, dict)
    assert raw

    rows, batches = load_human()
    assert len(rows) == len(raw)
    assert {"B1", "B2", "B3"} <= set(batches)
    assert all(batch for batch in batches.values())
