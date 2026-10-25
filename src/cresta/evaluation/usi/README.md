# User Simulation Index (USI)

This package implements the paper USI stack: atomic behavioral metrics, Dice
agreement, D1-D4, Eval, ECE, and the six-term score:

`USI = (D1 + D2 + D3 + D4 + Eval + (1 - ECE) * 100) / 6`

The default human dataset is
`datasets/usi/data/tau_bench_tasks_unified.json`. Override it with
`--human-file PATH`.

Run from the repository root:

```bash
# Inspect faithful atomic metrics on the human data.
python -m cresta.evaluation.usi.sim2real_metrics

# Reproduce the human inter-annotator Table 1 row and inspect simulator D1-D4.
python -m cresta.evaluation.usi.sim2real_full_table

# Inspect raw simulator atomic metrics. Strict is debug-only.
python -m cresta.evaluation.usi.sim2real_run_tau2 --mode strict path/to/results.json

# Compute full simulator USI where comparable rewards and surveys exist.
python -m cresta.evaluation.usi.sim2real_usi_full \
  --sim-glob "$TAU2_SIMULATIONS_DIR/*/airline/results.json" \
  --human-file datasets/usi/data/tau_bench_tasks_unified.json

# Run focused, offline tests.
pytest -q tests/repository/test_usi.py
```

`faithful` is the paper-compatible atomic metric mode. `strict` is available
only for inspecting cleaner atomic Accuse/Pivot/IDConf detections; it does not
define a second naturalness score.

`behMean` is only `mean(D1, D2, D3, D4)`. It is a partial behavioral diagnostic,
not USI. If Eval or ECE is unavailable, the full simulator command prints
`USI=N/A`; it never substitutes `behMean` or a five-term score.
