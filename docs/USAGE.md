# Usage Guide

How to run each part of the research pipeline. Commands assume you are at the repo
root with the venv active (`.venv/bin/python` or `uv run`). Set `OPENAI_API_KEY`
(or provider-specific shims) before any LLM call.

> Full reference for any script: `.venv/bin/python <script> --help`.
> Data note: framework sim inputs are self-contained in `datasets/`; experiment
> trajectory data must be available via `TAU2BENCH_DATA_ROOT` (decompress
> `main_trajectories.tar.zst` into the workspace).

---

## 0. Setup

```bash
uv sync --extra dev            # core + lint/test
uv sync --all-extras           # + voice / knowledge / gym / experiments
cp .env.example .env           # add API keys (LiteLLM — any provider)
uv run tau2 check-data         # verify install + data

# optional third-party deps for analysis scripts
uv pip install --index-url https://pypi.org/simple/ \
  scipy matplotlib scikit-learn thefuzz json_repair holidays gitpython \
  pymongo sqlparse nltk sacrebleu tiktoken pandas streamlit
```

Environment variables:
- `OPENAI_API_KEY` (or `ANTHROPIC_API_KEY` / shim tokens) — LLM calls
- `TAU2BENCH_DATA_ROOT` — workspace root for trajectories / sim output (default `../tau2bench-data`)
- `TAU2_DATA_DIR` — framework data dir (default `<repo>/datasets`)
- `PERSONA_DATA_DIR` — external persona datasets

---

## 1. Framework simulation (τ²-bench)

Run a customer-service agent against a user simulator on a domain.

```bash
# text, half-duplex
tau2 run --domain airline --agent llm_agent --agent-llm gpt-4.1 \
  --user user_simulator --user-llm gpt-4.1 --num-trials 1 --num-tasks 5

# all 5 domains: mock | airline | retail | telecom | banking_knowledge
# banking_knowledge needs a retrieval config:
tau2 run --domain banking_knowledge --retrieval-config bm25 \
  --agent llm_agent --agent-llm gpt-4.1 --user user_simulator --user-llm gpt-4.1
```

Output → `$TAU2BENCH_DATA_ROOT/artifacts/simulations/<timestamp>_<domain>_*`.
Browse with `tau2 view`. Each run writes `results.json` (reward, db_match,
status, full `messages` trajectory) + `trajectories.json`.

`tau2 intro` for a guided overview; `tau2 evaluate-trajs` to re-score old runs.

---

## 2. pgepa adversarial persona search

The core: find a user persona that maximizes agent difficulty (1 − reward)
while passing the validity judge (no fabrication / character-break / task-sabotage).

### τ²-bench domains (telecom / airline / retail / banking)

```bash
# dry-run smoke (no LLM cost, seconds)
.venv/bin/python experiments/persona_search/scripts/parallel_gepa.py \
  --seed-persona experiments/persona_search/seeds/telecom_seed.md \
  --dry-run --iterations 3 --minibatch 2 --max-tasks 2 --print-pareto

# real search (gpt-5.4 agent, telecom, 20 generations)
.venv/bin/python experiments/persona_search/scripts/parallel_gepa.py \
  --domain telecom --agent-model gpt-5.4 --reasoning-effort high \
  --seed-persona experiments/persona_search/seeds/telecom_seed.md \
  --iterations 20 --minibatch 10 --valset-size 30 \
  --out-dir $TAU2BENCH_DATA_ROOT/persona_opt/gepa_runs/my_telecom

# resume an existing run
.venv/bin/python experiments/persona_search/scripts/parallel_gepa.py \
  --run-dir $TAU2BENCH_DATA_ROOT/persona_opt/gepa_runs/my_telecom --resume
```

Key flags: `--domain`, `--agent-model`, `--reasoning-effort`, `--seed-persona`,
`--iterations`, `--minibatch`, `--valset-size`, `--out-dir`, `--run-dir`,
`--resume`, `--dry-run`, `--print-pareto`, `--min-dissimilarity`.

Adapter: `tau_persona_adapter.py` (wraps `tau2 run` via subprocess).

### ColBench (code)

```bash
# ColBench task data must be under vendor/sweet_rl/data/backend_tasks/
.venv/bin/python experiments/persona_search/scripts/gepa_runner_sweetrl.py \
  --data vendor/sweet_rl/data/backend_tasks/test_110_hard.jsonl \
  --run-dir $TAU2BENCH_DATA_ROOT/persona_opt/gepa_runs/sweetrl_run_001 \
  --iterations 20
```

Adapter: `sweetrl_persona_adapter.py` (uses `sweet_rl.utils.code_utils.check_correctness`).

### LiC (database / math / code / actions)

LiC uses its own `run_demo.py` instead of `tau2 run`. The adapter is
`vendor/lost_in_conversation/lic_persona_adapter.py`, driven via
`parallel_gepa.py` with the LiC adapter selected (see `cert_lic_test.py` for the
materialize + wrap_persona + demo subprocess path).

### VitaBench (delivery / instore / ota / cross_domain)

Adapter: `vendor/vitabench/scripts/vita_persona_adapter.py` (in-process
`vita.run.run_task`, rubric sliding-window evaluator). Driven via
`parallel_gepa.py` / `vita_pgepa.py`.

### Island model (parallel multi-island search)

```bash
.venv/bin/python experiments/persona_search/scripts/island_seen_pool.py   # shared seen-pool
# per-island parallel_gepa.py runs with distinct --seed-persona + --run-dir
```

**Artifacts written** per run: `candidates.json` (all personas + scores + parent),
`pgepa_state.json` (per-gen state, resume-able), `best_persona.md`,
`candidate_tree.html`, `run_log.{txt,json}`.

---

## 3. Certify (best persona vs no-persona on held-out test)

Quantify whether the searched persona actually makes the agent fail on the test
split (not seen during search).

### τ²-bench

```bash
.venv/bin/python experiments/persona_search/scripts/certify_gepa_candidates.py \
  --run-dirs $TAU2BENCH_DATA_ROOT/persona_opt/gepa_runs/my_telecom \
  --incumbent best --top-k 3 --num-trials 3 \
  --agent-llm gpt-5.4 --user-llm gpt-5.4 --effort high \
  --judge-model gpt-5.2 --judge-maj 3 \
  --max-severe-rate 0.05 --min-delta 0.05
```

Per-domain shortcuts: `cert_telecom_3model.py`, `cert_telecom_glm.py`,
`cert_telecom_opus.py`, `cert_sweetrl_test.py`, …

### LiC

```bash
.venv/bin/python experiments/persona_search/scripts/cert_lic_test.py \
  --domain database \
  --split-file vendor/lost_in_conversation/data/lic_split_database.json \
  --best-persona <best_persona.md from your pgepa run> \
  --trials 3 --agent-model accounts/fireworks/models/minimax-m3 --user-model gpt-5.4
```

**Artifact**: `cert_report.json` — `no_persona.avg_reward`,
`best_persona.avg_reward`, `delta_reward` (Δ > 0 = persona harder), `n_test`,
`trials`, `n_illegal`, `n_infra`.

---

## 4. Sim2Real naturalness (USI)

Score how realistic the persona-driven user simulator is vs human reference.

```bash
# full USI (D1–D4 behavioral Dice + ECE + Eval), ECE-only needs no LLM:
.venv/bin/python src/cresta/evaluation/usi/sim2real_usi_full.py \
  --sim-glob "$TAU2BENCH_DATA_ROOT/artifacts/simulations/*/airline/results.json" \
  --domain-pooled --calibrate

# ECE + Eval (LLM auto-survey):
.venv/bin/python src/cresta/evaluation/usi/sim2real_usi_full.py \
  --sim-glob '...' --domain airline --eval-model gpt-5.2

# naturalness vs a specific persona's sims (JSD / human-style):
.venv/bin/python experiments/persona_search/scripts/sim2real_naturalness.py \
  --sim-glob '...' --persona-file <best_persona.md>
```

Metrics: D1–D4 (behavioral Dice vs human), ECE (calibration), Eval MAE,
**USI** = mean(D1–D4, 1−ECE, Eval). Calibrated to paper human inter-annotator.
Reference data in `datasets/usi/`.

Related: `sim2real_full_table.py` (human table), `sim2real_metrics.py` (Dice),
`sim2real_naturalness_brinks.py`, `naturalness_reference.py`, `jsd_cross_domain.py`.

---

## 5. Validity / violation analysis

Ensure personas don't cheat (fabricate, character-break, sabotage task).

### Severe-rate validity judge (PPol format)

```bash
.venv/bin/python vendor/persona_policies/persona_policies/scripts/offline_validity_judge.py \
  --results <cert_results.json> --trajectories <trajectories.json> \
  --persona-json <candidates.json> --domain telecom \
  --judge-llm gpt-4.1 --severe-threshold 0.05 --max-workers 20 \
  --out <out.json> --env-file .env
```

Outputs `severe_rate` over 4 severe classes (fabrication, character_break, …).
Gate: `severe_rate ≤ 0.05` = valid persona.

### Intent preservation judge

```bash
.venv/bin/python experiments/baseline/scripts/judge_user_intent_preservation.py run \
  --results <results.json> --model gpt-15.2 --maj 3 --max-sims 50 \
  --out data/intent_eval/my_run --max-wait 1800
```

### pgepa dsv4 validity

```bash
.venv/bin/python experiments/baseline/scripts/pgepa_dsv4_validity.py \
  --results <dsv4_pgepa_cert_results.json>   # uses offline_validity_judge internally
```

### Roleplay probe (D1–D6)

```bash
.venv/bin/python experiments/baseline/scripts/roleplay_probe/roleplay_judge.py \
  --input <sim_dump.json> --judges gpt52 dsv4pro
.venv/bin/python experiments/baseline/scripts/roleplay_probe/roleplay_behavior_signals.py \
  --input <sim_dump.json>
```

---


## 6. NCUser (non-collaborative user behaviors)

Run the 5 NCUser behavior presets (none / emotional / tangential / fragment /
unavailable) on τ² domains.

```bash
# pilot: 10 tasks × 1 trial × 5 configs
.venv/bin/python experiments/persona_search/scripts/run_ncuser_telecom.py --mode pilot

# full: 54 tasks × 3 trials × 5 configs
.venv/bin/python experiments/persona_search/scripts/run_ncuser_telecom.py --mode full

# airline / retail variants:
.venv/bin/python experiments/persona_search/scripts/run_ncuser_airline.py --mode full
.venv/bin/python experiments/persona_search/scripts/run_ncuser_retail.py --mode full

# VitaBench delivery NCUser:
.venv/bin/python vendor/vitabench/run_ncuser_delivery.py
```

Each reports per-preset reward + Δ vs `none` baseline.

---

## 7. PPol (persona policies)

```bash
# eval a PPol-evolved persona policy on τ²
.venv/bin/python vendor/persona_policies/persona_policies/scripts/run_tau2_eval.py \
  --persona-json <policy.json> --domain airline --num-trials 3

# intent judge on PPol eval results
.venv/bin/python vendor/persona_policies/persona_policies/scripts/run_intent_judge.py \
  --results <results.json> --trajectories <trajectories.json> \
  --domain airline --judge-model gpt-5.2 --maj 3

# train the persona discriminator
.venv/bin/python vendor/persona_policies/persona_policies/scripts/train_discriminator.py

# VitaBench eval
.venv/bin/python vendor/persona_policies/persona_policies/scripts/run_vitabench_eval.py
```

---

## 8. tau-trait audit

```bash
.venv/bin/python vendor/tau-trait/auto_error_identification.py   # audit pipeline
.venv/bin/python experiments/persona_search/scripts/judge_tautrait.py  # tau-trait judge
```

---

## 9. Using published personas

Persona search outputs land in the run dir you pass to the runner
(`candidates.json` with every explored persona + score + parent,
`pgepa_state.json` for resume, `best_persona.md`). To evaluate a persona on
τ²-bench directly:

```bash
tau2 run --domain telecom --agent llm_agent --agent-llm gpt-5.4 \
  --user user_simulator --user-llm gpt-5.4 \
  --custom-guidelines "$(cat <run_dir>/best_persona.md)" --num-trials 3
```

