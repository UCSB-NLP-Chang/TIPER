# Repository architecture

The top level is organized by asset lifecycle, not by research topic. Code that
must be importable lives under `src/`; runnable studies under `experiments/`;
pinned third-party projects under `vendor/`.

## Top-level layout

| Area | Contents | May depend on |
|---|---|---|
| `src/tau2/` | τ²-bench simulation framework (agent/user/env/orchestrator/evaluator/metrics/runner/domains·knowledge·voice·gym/cli). Identical to upstream except ClaudeCodeUserSimulator is removed. | `datasets/` |
| `src/cresta/` | Research layer: `persona/` (optimization package — core/eval/mutation/search/diversity/induction/data_cleaning/analysis), `evaluation/usi/` (sim2real naturalness metrics), `evaluation/features.py`, `analysis/` | `tau2` |
| `src/experiments/` | Framework-adjacent experiments: `agentify_tau_bench/`, `hyperparam/`, `tau_voice/` | `tau2` |
| `experiments/baseline/scripts/` | Early attribution + `roleplay_probe/` (5) + `skill_trigger_eval/` + `backbone_capture/` + 18 root scripts (intent judge, pgepa validity, sim2real_llm_*, persona diversity) | `tau2`, `cresta` |
| `experiments/persona_search/` | pgepa search workspace: `configs/` (27), `seeds/` (20), `launchers/`, `scripts/` (122 .py + 87 .sh: GEPA engine, adapters, cert, ablation, judges, monitor), `scripts/viz/` (22) | `tau2`, `cresta`, `vendor/` |
| `vendor/` | 7 pinned third-party benchmarks (code-only): `persona_policies` (PPol), `vitabench`, `sweet_rl` (ColBench), `lost_in_conversation` (LiC), `NCUser`, `tau-trait`, `user_behavioral_divergence` | self-contained |
| `services/model_routing/` | 8 independently-launched model-routing shims (anthropic/openai → fireworks/vertex, vertex RR, probes) | installed packages |
| `datasets/` | Versioned inputs (173M): `tau2/` (all-domain tasks.json + db, self-contained), `cresta/`, `usi/` (human ref, 14M), `voice/` (18M) | — |
| `docs/` | Top-level guides (getting-started, CLI reference, USAGE) | — |
| `tests/` (64) | Framework, repository layout, lineage, USI structure tests | `src/` |
| `web/` | Leaderboard frontend | — |
| `benchmarks/` `tools/` `examples/` | Local benchmark specs, data manifest builders, agent examples | `src/` |

## External data workspace

Generated and proprietary data is not part of the Git checkout. By default it
lives at `../tau2bench-data/`:

```text
tau2bench-data/
  artifacts/          # simulations and benchmark outputs (40G trajectories)
  private/cresta/     # proprietary conversations
  persona_search/     # mutable state and historical runs
  caches/  history/
```

`TAU2BENCH_DATA_ROOT` changes the workspace root. More specific overrides:
`TAU2_DATA_DIR`, `CRESTA_DATA_DIR`, `CRESTA_PERSONA_DATA_DIR`,
`TAU2_ARTIFACTS_DIR`, `TAU2_SIMULATIONS_DIR`, `PERSONA_DATA_DIR`.

Trajectories are archived as `main_trajectories.tar.zst` (3.3G zstd -19 → 36.9G).
Decompress in the workspace root: `zstd -dc main_trajectories.tar.zst | tar -xf -`.

## Dependency rules

1. `src/tau2` never imports from `cresta`, experiments, services, or vendor.
2. Importable Cresta code lives under `src/cresta`; experiments contain only
   configuration, small fixtures, reports, and launchers.
3. Baselines depend on `tau2`, not persona-search internals.
4. Vendor projects stay self-contained and are addressed through explicit
   `sys.path` injection (each adapter injects its own vendor path from
   `__file__`).
5. Generated files are written through central path resolvers, never below
   `src/`, `experiments/`, `datasets/`, or `vendor/`.
6. Absolute workstation paths and compatibility symlinks are forbidden. All
   paths derive from `__file__` parents or env overrides.

## Persona runtime

`user_simulator` with `custom_guidelines` is the only supported persona injection
path. ClaudeCodeUserSimulator and its launchers are intentionally not in this repo.
The modular implementation in `src/cresta/persona/` is canonical; experiment
scripts (`tau_persona_adapter.py`, `sweetrl_persona_adapter.py`,
`lic_persona_adapter.py`, `vita_persona_adapter.py`) bridge it to each benchmark
via GEPA `EvaluationBatch`.

## Naturalness policy

USI is the sole supported naturalness calculation. Its implementation is
`src/cresta/evaluation/usi/` (`sim2real_usi_full.py` for full D1–D4 + ECE + Eval,
`sim2real_metrics.py` for behavioral Dice, `sim2real_full_table.py` for the
human-reference table). Historical JSD, LLM-realism, geometric P/R, and
naturalness-optimization experiments live as analysis scripts under
`experiments/` and the external history workspace.

## Cross-directory import injection

Because experiment scripts import modules from `src/cresta/evaluation/usi/`,
`experiments/baseline/scripts/`, and `vendor/*/`, each such script self-injects
the needed `sys.path` entries from `__file__` (walking up to the `pyproject.toml`
root). This keeps scripts runnable from any cwd without installing the research
layer. `pgepa_dsv4_validity.py` and the sim2real-LLM referrers are examples.
