"""Faithful User Simulation Index (USI) metrics."""


def usi(d1, d2, d3, d4, eval_, ece):
    """Compute the six-term paper USI."""
    from .sim2real_full_table import usi as _usi

    return _usi(d1, d2, d3, d4, eval_, ece)


def dice(a, b):
    """Compute Dice similarity for two atomic metric values."""
    from .sim2real_full_table import dice as _dice

    return _dice(a, b)


def metrics_for_interaction(turns):
    """Compute the paper's atomic metrics for one interaction."""
    from .sim2real_metrics import metrics_for_interaction as _metrics

    return _metrics(turns)


__all__ = ["dice", "metrics_for_interaction", "usi"]
