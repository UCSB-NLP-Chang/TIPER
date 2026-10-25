"""Shared behavioral feature extraction for persona induction and analysis.

This module intentionally exposes only the USI atomic metric vector. It does not
implement an alternative naturalness score.
"""

from __future__ import annotations

import numpy as np

from cresta.evaluation.usi import sim2real_metrics as metrics


def feature_matrix(turn_lists: list[list[str]]) -> np.ndarray:
    """Return one USI atomic-metric vector per non-empty interaction."""
    rows = []
    for turns in turn_lists:
        values = metrics.metrics_for_interaction(turns)
        if values:
            rows.append([values[name] for name in metrics.METRIC_ORDER])
    return np.asarray(rows, dtype=float)
