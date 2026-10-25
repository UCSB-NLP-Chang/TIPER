# Baseline evaluations

This directory contains experiment launchers and reports for evaluations that
do not optimize a Cresta persona.

- `scripts/`: baseline launchers, judges, and analysis utilities
- `reports/`: architecture and baseline experiment reports

Large published snapshots are stored externally at
`$TAU2BENCH_DATA_ROOT/results/published/tau2`.

The shared runtime remains at repository root:

```bash
uv run tau2 check-data
uv run tau2 run --domain mock --num-tasks 1 --num-trials 1
```

Generated rollouts are written to `$TAU2_SIMULATIONS_DIR`.
