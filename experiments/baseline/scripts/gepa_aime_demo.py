#!/usr/bin/env python3
"""GEPA demo on the AIME math dataset (default_adapter) with GPT-5.2.

Optimizes a single system-prompt component to maximize "exact answer contained
in response" on an AIME subset using GPT-5.2 for both task and reflection.

Run:  .venv/bin/python scripts/gepa_aime_demo.py
"""

import os


def load_env():
    """Populate os.environ from .env (project routes OpenAI via this key)."""
    if not os.path.exists(".env"):
        return
    for line in open(".env"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k, v.strip().strip('"').strip("'"))


def main():
    load_env()
    import gepa
    from gepa.examples.aime import init_dataset

    trainset, valset, _testset = init_dataset()

    trainset = trainset[:20]
    valset = valset[:10]

    seed_candidate = {
        "system_prompt": (
            "You are a helpful assistant. Solve the math problem. "
            "End your answer with '### <answer>'."
        )
    }

    print(f">>> trainset={len(trainset)} valset={len(valset)}")
    print(f">>> seed prompt: {seed_candidate['system_prompt']!r}\n")

    result = gepa.optimize(
        seed_candidate=seed_candidate,
        trainset=trainset,
        valset=valset,
        task_lm="openai/gpt-5.4",
        reflection_lm="openai/gpt-5.4",
        max_metric_calls=80,
        reflection_minibatch_size=5,
        display_progress_bar=True,
        seed=0,
    )

    print("\n" + "=" * 70)
    print("BEST VAL SCORE:", result.val_aggregate_scores[result.best_idx])
    print("=" * 70)
    print("BEST PROMPT:\n")
    print(result.best_candidate["system_prompt"])


if __name__ == "__main__":
    main()
