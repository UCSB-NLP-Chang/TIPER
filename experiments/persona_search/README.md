# Persona search experiments

- `configs/`: versioned task splits, representative sets, and initial library.
- `seeds/`: small, reproducible seed personas and trait cards.
- `launchers/`: supported commands that call the installed `cresta` package.

Mutable state defaults to
`$TAU2BENCH_DATA_ROOT/persona_search/state`; simulations default to
`$TAU2BENCH_DATA_ROOT/artifacts/simulations`.

Run an offline smoke test:

```bash
DRY_RUN=1 ITERATIONS=1 MAX_TASKS=2 \
  bash experiments/persona_search/launchers/run_pgepa.sh
```

Prepare one isolated experiment per supported tau2 domain:

```bash
.venv/bin/python experiments/persona_search/setup_benchmark_experiments.py \
  --run-tag my_matrix
```

The command validates the domain-specific split and seed files, then writes a
`plan.json` and executable launcher for telecom, retail, airline, and banking
under `$CRESTA_PERSONA_DATA_DIR/benchmark_setups/my_matrix/`. It never copies API
keys into generated files.

Validate the entire generated matrix without API calls:

```bash
.venv/bin/python experiments/persona_search/setup_benchmark_experiments.py \
  --run-tag matrix_smoke \
  --iterations 1 \
  --execute dry-run
```

Use `--domains telecom retail`, for example, to prepare a subset. Add
`--execute live` only when `OPENAI_API_KEY` is exported and the selected models
and run size are intentional.
