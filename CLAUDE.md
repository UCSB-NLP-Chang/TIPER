# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> See also: `AGENTS.md` — the primary reference for architecture, patterns, and conventions. Read it before making changes.

## Setup

```bash
uv sync --extra dev            # minimum for running tests and linting
uv sync --all-extras           # everything (voice, knowledge, gym, experiments)
cp .env.example .env           # add API keys
uv run tau2 check-data         # verify installation
```

Voice features additionally need: `brew install portaudio ffmpeg`

## Common Commands

```bash
make test          # core tests (no voice/knowledge/gym)
make test-voice    # voice + streaming tests
make test-knowledge  # banking_knowledge tests
make test-gym      # gymnasium tests
make test-all      # all tests

# Run a single test file
uv run pytest tests/test_agent.py
uv run pytest tests/test_domains/test_airline

# Skip tests that require live APIs
uv run pytest -m "not full_duplex_integration"

make lint          # ruff check
make format        # ruff format
make check-all     # lint + format (required before committing)
```

## Architecture

The framework simulates agent↔user↔environment interactions. The runner wires them together; the orchestrator drives the conversation loop.

```
tau2 CLI (cli.py)
  └─> runner/ (batch.py, build.py, simulation.py)
        ├─> agent/        — system under test (LLMAgent, DiscreteTimeAudioNativeAgent)
        ├─> user/         — user simulator (UserSimulator; CC-user variants intentionally excluded)
        ├─> environment/  — domain tools + DB
        └─> orchestrator/ — drives the loop (half-duplex or full-duplex)

evaluator/   — scores a completed simulation run
metrics/     — aggregates scores across runs
registry.py  — global lookup for agents, domains, tasks, users
config.py    — single source of truth for default values
```

### Research layer & experiments

This is a code-first research monorepo on top of τ²-bench (adversarial persona
search). See **README.md §Repository Structure** and **ARCHITECTURE.md** for full
detail. Key areas beyond the framework:

- `src/cresta/` — persona optimization package (`persona/`) + sim2real USI metrics (`evaluation/usi/`)
- `experiments/persona_search/scripts/` — pgepa search engine, per-benchmark adapters (`{tau,sweetrl,lic,vita}_persona_adapter.py`), cert, ablation, judges, monitor
- `vendor/` — 7 pinned third-party benchmarks (PPol, vitabench, sweet_rl/ColBench, LiC, NCUser, tau-trait, user_behavioral_divergence)

All experiment script paths are portable: `__file__` derivation + `TAU2BENCH_DATA_ROOT`
env override. Cross-directory imports are self-injected. Trajectories (40G) are not
in git — decompress `main_trajectories.tar.zst` into the data workspace.

## Key Patterns

**Registry**: Every agent, domain, and user simulator must be registered in `registry.py` to be accessible from the CLI or `tau2 run`.

**Domain layout**: `src/tau2/domains/<name>/` contains `data_model.py`, `tools.py`, `environment.py`; data lives in `datasets/tau2/domains/<name>/` (self-contained, 173M). The `base` task split is the default for evaluation.

**Half-duplex vs full-duplex**: `HalfDuplexAgent.generate_next_message()` for turn-based; `FullDuplexAgent.get_next_chunk()` for tick-based streaming. User simulators follow the same pattern.

**`banking_knowledge` domain**: Requires `--retrieval-config` (e.g., `bm25`, `qwen_embeddings`). Offline configs need no API keys; `openai_embeddings*` requires `OPENAI_API_KEY`; `terminal_use*` requires `npm install -g @anthropic-ai/sandbox-runtime@0.0.23`.

## Gotchas

- `config.py` is the single source of truth for defaults — import from there.
- `langfuse` and `redis` are optional; install manually if `USE_LANGFUSE=True` or `LLM_CACHE_ENABLED=True` with redis.
- Notebooks (`*.ipynb`) are excluded from ruff.
- The `base` task split is the default for evaluation; `train`/`test` are for RL experiments.
- Pre-commit hook runs `make check-all` — fix ruff issues before committing.
