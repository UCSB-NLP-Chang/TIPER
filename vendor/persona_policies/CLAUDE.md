# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

**Persona Policies (PPol)** evolves realistic, diverse user-simulator personas for LLM agent benchmarks. It is a control layer on top of **τ²-bench (tau2)**: it appends a natural-language "persona policy" block to the tau2 user simulator's system prompt to vary *how* the simulated user communicates, while leaving task goals, private facts, rewards, and tools unchanged. Personas are optimized via evolutionary program search (OpenEvolve) against two objectives — human-likeness and behavioral coverage.

This repo is a **standalone research project**. τ²-bench is an external dependency that is *not* vendored here — install it from its official source (`pip install -e /path/to/tau2-bench`) before running anything. Do not confuse this repo with the neighboring `tau2bench_cc` fork; this project does **not** use `ClaudeCodeUserSimulator` — it subclasses tau2's `UserSimulator` directly (see `injector.py`).

## Setup

```bash
conda env create -f environment.yml && conda activate persona-policies
pip install -e /path/to/tau2-bench        # external dep, not included
pip install -r persona_policies/requirements.txt
python persona_policies/evolution/install_openevolve_litellm_pth.py   # once per venv, routes OpenEvolve mutations through LiteLLM

export OPENROUTER_API_KEY=...            # Gemini/Gemma models (openrouter/... prefix)
export GEMINI_API_KEY=...                # gemini/... LiteLLM ids
export AWS_REGION_NAME=us-west-2         # bedrock/... models
```

## Common commands

Run everything from the **repo root** (modules are imported as `persona_policies.*` via repo-root `sys.path` insertion). Tests/evolution break if run from inside `persona_policies/`.

```bash
# Full pipeline (steps: data, baseline, discriminator, evolve, benchmark)
python persona_policies/run_pipeline.py --iterations 70 --domain retail
./persona_policies/run.sh                                # shell version, clean-tree + resume-friendly

# Individual stages
python persona_policies/scripts/collect_baseline.py --n 100
python persona_policies/scripts/train_discriminator.py
python persona_policies/evolution/run_evolution.py --iterations 70
python persona_policies/evolution/run_evolution.py --iterations 100 --resume    # from latest checkpoint
python -m persona_policies.benchmark --best-program persona_policies/outputs/training_v1/openevolve/best/best_program.py
python -m persona_policies.benchmark --best-by-val-avg --n-personas-list 5,8,10  # pick checkpoint by mean val score
python -m persona_policies.benchmark --baseline no_persona        # or seed_initial_generator / direct_llm_personas

# Tests
uv run pytest persona_policies/tests/test_fingerprinting.py        # if uv is used
pytest persona_policies/tests/test_fingerprinting.py               # only test file; requires pytest>=8
```

There is no Makefile, lint config, or formatter wired up here — formatting/linting is not part of this repo's workflow. The only tests live under `persona_policies/tests/` (fingerprinting sanity checks); they need no API keys.

## Architecture

### The two-stage evolved generator

`persona_policies/evolution/initial_generator.py` implements `G(c, D, N)` — the **program that OpenEvolve mutates**. It is the heart of the project. OpenEvolve rewrites this file's LLM prompts across iterations; it is both source code and the unit of evolution.

- `c`: formatted tau2 user scenario for one task
- `D`: `DIVERSITY_AXES` — behavioral axes defined in this same file
- `N`: number of personas to generate

Generation is two-stage: `generate_population(...)` jointly proposes `N` population members with axis placements, then `expand_personas_parallel(...)` expands each into full roleplay instructions (one Stage-2 LLM call per member, parallelized).

### The evolution loop

`evolution/fitness.py` is OpenEvolve's `evaluate(program_path)` callback. One OpenEvolve iteration = one training batch step:

1. Load the evolved `G(c, D, N)` from `program_path`.
2. Sample a random minibatch of **train-split** tau2 tasks (split internally into train slice + held-out val slice via `val_fraction`).
3. For each task: build context `c`, call `G` to get `N` personas, run one tau2 episode per `(persona, task)` pair in parallel (`parallel_episode_workers`).
4. Score with curriculum-aware weights → `combined_score`. OpenEvolve uses train-only score to decide elites.
5. Build an `evaluator_reflection` artifact (scalar metrics + sampled dialogue excerpts + persona text + compact fingerprints) and return it as dynamic feedback to the mutator.
6. **Validation is monitor-only**: when the on-disk elite changes, a daemon thread (no `join`) runs full val and appends to `results/val_curve.jsonl`. Training never waits on val.

OpenEvolve then mutates `initial_generator.py` using static guidance (`evolution/templates/system_message.txt`), the dynamic reflection artifact, and one sampled `{mutation_angle}` slotted into `evolution/templates/diff_user.txt`. Mutation angles are defined under `prompt.template_variations.mutation_angle` in `openevolve_config.yaml`.

### Two fitness objectives (in `evaluator.py`)

- **Human-likeness (HL)**: mean `P(human)` per episode from a trained Random Forest discriminator (`discriminator.py`) over 19 behavioral regex features (`fingerprinting.py`). The discriminator pkl under `outputs/reference_data/` is **required** at evaluator startup.
- **Behavioral coverage (B_cover)**: per-task Chamfer distance between generated persona fingerprints and the train-split human reference cloud, normalized to [0, 1].
- **Combined**: `score = lambda_h * HL + lambda_b * B_cover`. Training uses curriculum-adjusted weights (λ_b ramps with current training `N`); validation/benchmark use configured weights directly.

`fingerprinting.py` also computes an aggregate-first Sørensen–Dice score, but it is **diagnostic only** — it never enters fitness. Reflection intentionally excludes per-feature Dice dimensions so the mutator does not game brittle proxy features (word count, capitalization, punctuation).

### Persona injection & episode running (`injector.py`)

`inject_persona_into_system_prompt()` appends the `PERSONA_INJECTION_TEMPLATE` block after tau2's full base system prompt. `_persona_user_class()` builds a cached `PersonaPolicyUserSimulator(UserSimulator)` subclass that overrides the `system_prompt` property so persona text flows into tau2's `get_init_state`.

`TaubenchEpisodeRunner` runs programmatic tau2 episodes: builds a `TextRunConfig`, swaps in `PersonaPolicyUserSimulator`, and calls `run_simulation(orchestrator)` with `EvaluationType.ALL_WITH_NL_ASSERTIONS`. It loads tasks via `get_tasks(domain, task_split_name)` so train/test splits match `split_tasks.json`. Supports combined domains (e.g. `retail_airline`).

### tau2 wiring (`taubench_setup.py`)

`ensure_taubench_importable()` sets `TAU2_DATA_DIR` and prepends tau2's `src/` to `sys.path`. Must be called before importing any `tau2.*` module. `apply_tau2_bedrock_judge_overrides()` patches tau2's NL-assertion and env-interface judge defaults (tau2 defaults to OpenAI `gpt-4.1`; this repo strips `OPENAI_API_KEY` for Bedrock-only runs, so without the patch `ALL_WITH_NL_ASSERTIONS` intermittently fails).

### Configuration (`config.py`)

`PersonaPoliciesConfig` is the single source of truth. It computes all artifact paths from a few domain/model fields (`baseline_artifact_stem`, `human_fingerprints_filename`, `discriminator_filename`) and calls `refresh_domain_artifact_paths()` when domain/model change. Several fields are overridable via `PERSONA_POLICIES_*` env vars (version, domain, user/agent model, val_fraction) — `run_pipeline.py`/`run_evolution.py` set these before constructing the config. `ensure_output_dirs()` creates the canonical `outputs/` tree and warns on stray top-level dirs.

### SFT case study (`sft_experiments/`)

Separate LoRA fine-tuning pipeline (Gemma-4-31B) comparing default-only vs default+PPol tau2 rollouts. Three domains: `retail`, `airline`, `retail_airline`. Collect raw JSONL trajectories → `build_sharegpt_sft_data.py` filters to `reward >= 1.0` and writes ShareGPT for LLaMA-Factory → `train_lora.py` launches `llamafactory-cli train`. Requires external deps: `tau2-bench`, `tau-trait` (OOD user suite), `LLaMA-Factory`, plus env vars `TAU2_BENCH_ROOT`, `LLAMA_FACTORY_DIR`, `HF_TOKEN`. See `sft_experiments/README.md`.

## Output layout

All durable artifacts live under `persona_policies/outputs/` (gitignored, created at runtime):

```
reference_data/   baseline_*_results/fingerprints/trajectories.json, human_fingerprints_<domain>.json, discriminator_<domain>_<agent>_user_<user>.pkl
training_<ver>/   openevolve/{checkpoints,best/best_program.py,logs}, simulations/iter_NNNN/, validation/iter_NNNN/, results/{train,val}_curve.jsonl + evolution_best.json
testing/          benchmark runs: log.json, trajectories.json, fingerprints.json, summary.txt, *.png
```

## Gotchas

- **Run from repo root**, not from inside `persona_policies/`. Scripts insert the repo root into `sys.path` themselves, but CWD-relative paths and `python -m persona_policies.benchmark` assume repo root.
- `train_max_steps_per_episode` (training) and `max_turns_per_episode` (val/benchmark) both cap tau2 **orchestrator steps**, not clean human/assistant turns — user, assistant, and tool/environment steps all count.
- Discriminator pkl and baseline JSONs are keyed by **both** agent model and user simulator model in their filenames. Changing either model invalidates the cache — re-run `collect_baseline` then `train_discriminator` for the new model combo.
- Curriculum changes only training batches: default schedule is `N=5` (epoch 1), `N=8` (epoch 2), `N=10` (epoch 3+). Validation and benchmark always use full `n_personas`.
- `run.sh` wipes `outputs/` unless `SKIP_PIPELINE_CLEAN=1`; use that env var to keep checkpoints for `--resume`.
- `evolution/initial_generator.py` is mutated in place by OpenEvolve during a run — treat its current contents as a checkpoint, not a hand-edited source of truth (the committed version is the seed).
- See `persona_policies/docs/taubench_map.md` for the precise tau2 internals (system prompt assembly, `split_tasks.json`, `get_tasks`, reward semantics) that this project hooks into.
