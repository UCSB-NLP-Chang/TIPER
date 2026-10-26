# TIPER — Adversarial Persona Search Research Monorepo

> Code for the paper **[Uncovering Robustness Weaknesses in LLM Agents via Persona
> Evolution with Task-Integrity Constraints](https://arxiv.org/pdf/2610.16155)**
> ([arXiv:2610.16155](https://arxiv.org/abs/2610.16155)).
>
> A code-first research monorepo built on the [τ²-bench](https://arxiv.org/abs/2506.07982)
> simulation framework. Reusable packages in `src/`, runnable studies in `experiments/`,
> versioned inputs in `datasets/`, third-party benchmarks in `vendor/`, and
> model-routing shims in `services/`.

[![python](https://img.shields.io/badge/Python-3.12%2B-blue.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![arXiv](https://img.shields.io/badge/cs.AI-arXiv%3A2610.16155-B31B1B.svg?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2610.16155)

## Overview

This repo implements **TIPER**: adversarial user-persona search — finding user
simulator personas that reliably make a target agent fail, while staying
realistic (natural, non-fabricating). The core engine is **pgepa** (parallel-generational GEPA), a
genetic search that mutates personas via an LLM proposer and selects on a
difficulty objective (1 − agent reward), gated by a validity judge that rejects
task-sabotaging / fabricating personas.

The research pipeline runs across **4 benchmark families** (τ²-bench 4 domains,
LiC, ColBench, VitaBench) and **9+ target agents** (gpt-5.4, sonnet-5, dsv4-flash,
glm-5.2, minimax-m3, haiku, muse, qwen, gpt-oss).

> **ClaudeCodeUserSimulator is intentionally excluded** from this repo. Only the
> standard `user_simulator` + `custom_guidelines` persona injection path is supported.

## Repository Structure

### `src/tau2/` — τ²-bench simulation framework

The upstream framework, identical to sierra-research/tau2-bench except the CC-user
simulator is removed. It wires agent ↔ user ↔ environment and scores outcomes.

| Subpackage | Role |
|---|---|
| `agent/` | Agents under test — `LLMAgent` (text half-duplex), `DiscreteTimeAudioNativeAgent` (voice full-duplex) |
| `user/` | User simulators — `UserSimulator` (the only one here; CC-user variants excluded) |
| `environment/` | Domain tools + database the agent acts on |
| `orchestrator/` | Drives the conversation loop — half-duplex (turn-based) or full-duplex (tick-streaming) |
| `evaluator/` | Scores a completed simulation run against expected actions / DB state |
| `metrics/` | Aggregates scores across runs (reward, DB match, pass^k) |
| `runner/` | `build.py` / `batch.py` / `simulation.py` — assembles and runs a batch of sims |
| `domains/` | 5 domains: `mock`, `airline`, `retail`, `telecom`, `banking_knowledge` (each has `data_model.py`, `tools.py`, `environment.py`) |
| `knowledge/` | RAG retrieval pipeline configs for `banking_knowledge` (bm25 / embeddings / terminal-use) |
| `voice/` | Full-duplex audio-native adapters (OpenAI / Gemini / xAI realtime) |
| `gym/` | Gymnasium-compatible RL environment interface (train/test splits) |
| `cli.py` | The `tau2 run / view / evaluate-trajs / check-data` CLI |
| `registry.py` | Global lookup for agents, domains, tasks, users |
| `config.py` | Single source of truth for defaults |

### `src/cresta/` — research layer

The reusable research code, packaged and importable (`import cresta`). Depends on
`tau2` only.

| Subpackage | Role |
|---|---|
| `persona/core/` | Persona data model, candidate representation, materialization |
| `persona/eval/` | Per-domain evaluation (`eval_persona.py`, `_run_domain_with_retry`) |
| `persona/mutation/` | LLM-driven persona mutation proposer (edit-rate controlled) |
| `persona/search/` | GEPA loop integration, Pareto-front selection, reflective dataset |
| `persona/diversity/` | Persona diversity maintenance (dispersion, frontier, JSD) |
| `persona/induction/` | Seed-persona induction from failure trajectories |
| `persona/data_cleaning/` | Persona text cleaning / normalization |
| `persona/analysis/` | Persona behavior analysis utilities |
| `evaluation/usi/` | **Sim2Real** naturalness metrics — `sim2real_metrics.py` (D1–D4 behavioral Dice), `sim2real_full_table.py`, `sim2real_usi_full.py` (full USI: D1–D4 + ECE + Eval), `sim2real_run_tau2.py`. USI is the canonical naturalness score. |
| `evaluation/features.py` | Shared feature extraction |
| `analysis/` | Higher-level analysis tools |

### `src/experiments/` — framework-adjacent experiments

`agentify_tau_bench/`, `hyperparam/`, `tau_voice/` — small framework-level
experiments kept close to `src/`.

### `experiments/` — runnable research studies

All experiment launchers and analysis scripts. **Paths are portable**
(`__file__` derivation + `TAU2BENCH_DATA_ROOT` env); cross-directory imports are
self-injected.

#### `experiments/baseline/scripts/`

Early attribution, analysis, and probe scripts:

| Group | Scripts | Role |
|---|---|---|
| `roleplay_probe/` (5) | `roleplay_judge`, `roleplay_behavior_signals`, `roleplay_report`, `aggregate_personas`, `persona_stats` | Adversarial persona role-play probe benchmark — D1–D6 judge + behavior signals |
| `skill_trigger_eval/` | skill-trigger evaluation scripts | Skill-trigger analysis |
| `backbone_capture/` | backbone capture scripts | LLM backbone API capture |
| root (18) | `judge_user_intent_preservation`, `pgepa_dsv4_validity`, `ppol_to_sim2real`, `sim2real_llm_metrics/calibrate/pr/viz`, `persona_behavior_jsd`, `persona_dispersion`, `persona_frontier_diversity`, `run_hybrid_probe`, `run_no_persona_val25`, `minimax_routing`, … | Violation/intent preservation judge, pgepa validity, PPol→sim2real bridge, sim2real LLM metrics family, persona diversity (JSD/dispersion/frontier), baseline launchers, model-routing helpers |

#### `experiments/persona_search/`

The main pgepa search workspace.

| Subdir | Role |
|---|---|
| `configs/` (27) | Per-experiment YAML configs (domain, model, split, hyperparams) |
| `seeds/` (20) | Seed personas for each domain/model |
| `launchers/` | Top-level pipeline launchers (`run_full_pipeline`, `run_loop`, `run_round`) |
| `scripts/` (122 .py + 87 .sh) | **pgepa engine + adapters + cert + ablation + judges + monitor** (see below) |
| `scripts/viz/` (22) | Visualization scripts (PCA coverage, manifold P/R, difficulty curves, d1–d4 plots) |

**Key `scripts/` files:**

| Script | Role |
|---|---|
| `parallel_gepa.py` | **pgepa core** — generational GEPA driver, per-instance Pareto, K children/gen, `--resume` from `pgepa_state.json` |
| `gepa_runner.py` | τ²-bench domain runner (telecom/airline/retail/banking) |
| `gepa_runner_sweetrl.py` | ColBench code runner |
| `crossdomain_gepa.py` | Cross-domain search |
| `island_seen_pool.py` | Island-model: isolated candidate pools + file-locked shared seen-pool |
| `tau_persona_adapter.py` | τ²-bench GEPA adapter (`EvaluationBatch`) — wraps `tau2 run` via subprocess |
| `sweetrl_persona_adapter.py` | ColBench GEPA adapter — uses `sweet_rl.utils.code_utils.check_correctness` |
| `cert_lic_test.py` / `cert_sweetrl_test.py` / `cert_telecom_*` | Per-benchmark certification (best persona vs no-persona on test split) |
| `certify_gepa_candidates.py` | Generic cert driver |
| `run_ncuser_{telecom,airline,retail}.py` | NCUser non-collaborative-user runs |
| `run_enriched_ablation_v2.py` / `run_model_ablation_vita.py` | Ablation studies (no-judge / no-gate / no-outcome / model ablation) |
| `failure_taxonomy_all_domains.py` | Error-type / failure taxonomy |
| `judge_persona.py` / `judge_tautrait.py` / `cross_judge_glm.py` / `judge_calibration_test.py` | Judge scripts |
| `sweetrl_offline_judge.py` / `sweetrl_judge_at_k_trajectories.py` | ColBench offline judging |
| `monitor*` / `pgepa_monitor.py` / `pgepa_lineage_tree.py` | Live run monitoring + lineage tree |
| `fullset_build.py` / `fullset_summary.py` | Full-set assembly + summary |
| `sim2real_naturalness*.py` / `naturalness_reference.py` / `score_naturalness.py` | Naturalness scoring (JSD, human-style, USI) |
| `run_*_nopersona_test.py` / `*_nopersona_test.py` | No-persona baselines per benchmark |

### `vendor/` — pinned third-party benchmarks (code-only)

Seven self-contained projects, addressed through explicit `sys.path` injection.
Data/trajectories are excluded (code-only); each has its own README.

| Project | Bench | Role |
|---|---|---|
| `persona_policies/` (PPol) | τ² + VitaBench | Persona-policy optimization (evolution, discriminator, intent judge, offline validity judge, τ²/VitaBench eval runners) |
| `vitabench/` | delivery / instore / ota / cross_domain | VitaBench domains + `vita_persona_adapter.py` + NCUser delivery + `vita_offline_judge.py` (data 44M included) |
| `sweet_rl/` (ColBench) | code / database / math | ColBench RL environment + `check_correctness` scorer (task data **not** included — see Data) |
| `lost_in_conversation/` (LiC) | database / math / code / actions | LiC multi-turn instruction benchmark + `lic_persona_adapter.py` + `tasks/` + `judge_lic_conversations.py` (split JSONs included, trajectories excluded) |
| `NCUser/` | telecom / airline (τ²) + multiwoz | Non-Collaborative User paper source — 5 non-cooperative behaviors |
| `tau-trait/` | trait evaluation | τ-trait project incl. `auto_error_identification.py` (audit pipeline) |
| `user_behavioral_divergence/` | user behavior | User behavioral divergence analysis |

### `services/model_routing/` — model routing shims (8)

Independently launched proxies that route LLM calls to different providers/models:

| Script | Role |
|---|---|
| `anthropic_to_fireworks_proxy.py` | Anthropic API → Fireworks |
| `anthropic_to_openai_proxy.py` | Anthropic API → OpenAI |
| `anthropic_to_vertex_proxy.py` | Anthropic API → Vertex AI |
| `openai_to_fireworks_proxy.py` | OpenAI API → Fireworks |
| `vertex_rr_shim.py` | Vertex round-robin + backoff |
| `routine.py` | Shared routing routine |
| `list_vertex_models.py` / `probe_vertex_models.py` | Vertex model discovery |

### `datasets/` — versioned inputs

| Subdir | Role |
|---|---|
| `tau2/` (140M) | All-domain task definitions: `domains/{airline,retail,telecom,banking_knowledge,mock}/tasks.json` + `db.json` + policy. **Self-contained — framework sim needs nothing else.** |
| `usi/` (14M) | Sim2Real human reference data (inter-annotator, survey) |

### `docs/` — documentation

- Top-level: `getting-started.md`, `cli-reference.md`, `evaluation.md`,
  `interaction-metrics.md`, `USAGE.md` (concrete commands for every feature)

### `tests/` (64) · `web/` · `benchmarks/` · `tools/` · `examples/`

- `tests/` — framework, repository layout, lineage, and USI structure tests
- `web/` — leaderboard frontend
- `benchmarks/` — local benchmark specs + provenance
- `tools/` — data manifest builders
- `examples/` — agent examples

## Research Pipeline

The end-to-end adversarial persona research loop:

```
1. Seed persona        experiments/persona_search/seeds/  +  src/cresta/persona/induction/
2. pgepa search        parallel_gepa.py  +  {tau,sweetrl,lic,vita}_persona_adapter.py
   ↓ mutates personas via LLM proposer; selects on difficulty (1−reward), validity-gated
   ↓ writes candidates.json + pgepa_state.json + best_persona.md
3. Certify             certify_gepa_candidates.py / cert_*_test.py
   ↓ best persona vs no-persona on held-out test split, avg_reward@k, pass^k, Δ
   ↓ writes cert_report.json
4. Naturalness         src/cresta/evaluation/usi/  (sim2real_*)  +  sim2real_naturalness*.py
   ↓ USI (D1–D4 + ECE + Eval) vs human; JSD; manifold P/R
5. Validity / violation  offline_validity_judge.py, judge_user_intent_preservation.py,
   ↓ roleplay_judge.py, pgepa_dsv4_validity.py
   ↓ severe_rate (fabrication / character_break / ...), intent_preserved, D1–D6
```

**Unified validity gate** (all 4 adapters — tau / vita / lic / sweetrl): failed
non-infra trials are audited by a shared severe judge (`src/cresta/persona/eval/severe_judge.py`,
gpt-5.2, **maj=3 strict-majority votes** per trial) on 4 categories (fabricated_info /
refused_to_engage / contradictory / character_break). Task-level severe = ANY trial severe;
task-level timeout = ALL trials timeout. Two gates zero the whole candidate:
`severe_rate = severe_tasks / n_noninfra > 0.05` and `timeout_rate >= 0.15`. On LiC the
shard-transmission rule signal feeds `timeout_safety` (same 0.15 gate) instead of severe —
severe is judge-only. Judge errors fail open (counted as genuine failures). VitaBench's
rubric reward scorer (`evaluator_llm`) is likewise unified on gpt-5.2.

Run a single search (τ² telecom, gpt-5.4 agent):

```bash
.venv/bin/python experiments/persona_search/scripts/parallel_gepa.py \
  --domain telecom --agent-model gpt-5.4 --iterations 20 --out-dir data/persona_opt/gepa_runs/my_run
```

## Quick Start

### 1. Install

```bash
uv sync --extra dev            # core + lint/test
uv sync --all-extras           # + voice / knowledge / gym / experiments
cp .env.example .env           # add API keys
uv run tau2 check-data         # verify installation
```

Python `>=3.12, <3.14`. Voice needs `brew install portaudio ffmpeg`.

### 2. Set up API keys

```bash
cp .env.example .env   # LiteLLM — any provider works
```

Data workspace defaults to `../tau2bench-data/`. Override with `TAU2BENCH_DATA_ROOT`;
finer overrides: `TAU2_DATA_DIR`, `CRESTA_DATA_DIR`, `TAU2_SIMULATIONS_DIR`,
`PERSONA_DATA_DIR`. Experiment scripts read sim data via `TAU2BENCH_DATA_ROOT`.

### 3. Run an evaluation

```bash
tau2 run --domain airline --agent llm_agent --agent-llm gpt-4.1 \
  --user user_simulator --user-llm gpt-4.1 --num-trials 1 --num-tasks 5
```

Results → `$TAU2BENCH_DATA_ROOT/artifacts/simulations/`. `tau2 view` to browse.

## Data

This repo is **code-only** for experiment outputs. Framework sim inputs
(`datasets/tau2/` 173M) are self-contained.

ColBench task data (`vendor/sweet_rl/data/backend_tasks/`) is not vendored —
obtain from the [sweet_rl/ColBench](https://github.com/MAGA-Lian/sweet_rl)
project and place under `vendor/sweet_rl/data/`.

## Documentation

| Doc | Description |
|---|---|
| [Usage Guide](docs/USAGE.md) | **Concrete commands for every feature** (sim, pgepa, cert, USI, validity, case study, NCUser, PPol, audit, ablation) |
| [Getting Started](docs/getting-started.md) | Install, API keys, first run |
| [CLI Reference](docs/cli-reference.md) | All `tau2` commands |
| [Evaluation](docs/evaluation.md) | Task schema, reward, action correctness |
| [Architecture](ARCHITECTURE.md) | Dependency rules, data workspace, persona runtime |
| [Agent Dev Guide](src/tau2/agent/README.md) | Build your own agent |
| [Domains](src/tau2/domains/README.md) | Domain structure & data format |

## Citation

If you use this repo or the persona-evolution methodology, please cite the TIPER paper:

```bibtex
@misc{xie2026tiper,
  title={Uncovering Robustness Weaknesses in LLM Agents via Persona Evolution with Task-Integrity Constraints},
  author={Yutao Xie and Jingbo Yang and Guanyu Yao and Xiaoyu Yang and Duo Ding and Shiyu Chang},
  year={2026},
  eprint={2610.16155},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://arxiv.org/abs/2610.16155}
}
```
