#!/usr/bin/env python3
"""GEPA prompt optimization on AIME 2025 with default_adapter + GPT-5.4.

Dataset: MathArena/aime_2025 (train from AI-MO/aimo-validation-aime, test from MathArena)
Patches DefaultAdapter.evaluate to use batch_completion with error handling.

Run:  .venv/bin/python scripts/gepa_aime2025.py
"""

import os
import random


def load_env():
    if not os.path.exists(".env"):
        return
    for line in open(".env"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k, v.strip().strip('"').strip("'"))


def load_aime2025():
    from datasets import load_dataset

    train_split = []
    train_load = load_dataset("AI-MO/aimo-validation-aime", "default", split="train")
    for item in train_load:
        train_split.append(
            {
                "input": item["problem"],
                "additional_context": {"solution": item["solution"]},
                "answer": str(item["answer"]),
            }
        )
    random.Random(0).shuffle(train_split)

    test_split = []
    test_load = load_dataset("MathArena/aime_2025", "default", split="train")
    for item in test_load:
        test_split.append(
            {
                "input": item["problem"],
                "additional_context": {},
                "answer": str(item["answer"]),
            }
        )

    mid = len(train_split) // 2
    return train_split[:mid], train_split[mid:], test_split


def patch_default_adapter():
    """Patch DefaultAdapter to handle ContentPolicyViolationError in batch_completion."""
    from typing import cast

    from gepa.adapters.default_adapter.default_adapter import (
        DefaultAdapter,
        EvaluationBatch,
    )

    _orig_evaluate = DefaultAdapter.evaluate

    def safe_evaluate(self, batch, candidate, capture_traces=False):
        outputs = []
        scores = []
        objective_scores = []
        trajectories = [] if capture_traces else None

        system_content = next(iter(candidate.values()))

        litellm_requests = []
        for data in batch:
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": data["input"]},
            ]
            litellm_requests.append(messages)

        if isinstance(self.model, str):
            raw_responses = self.litellm.batch_completion(
                model=self.model,
                messages=litellm_requests,
                max_workers=self.max_litellm_workers,
                **self.litellm_batch_completion_kwargs,
            )
            responses = []
            for resp in raw_responses:
                try:
                    responses.append(resp.choices[0].message.content.strip())
                except (AttributeError, IndexError, TypeError):
                    responses.append("[no response]")
        else:
            responses = [self.model(messages) for messages in litellm_requests]

        for data, assistant_response in zip(batch, responses, strict=True):
            eval_result = self.evaluator(data, assistant_response)
            output = {"full_assistant_response": assistant_response}
            outputs.append(output)
            scores.append(eval_result.score)
            objective_scores.append(eval_result.objective_scores)
            if trajectories is not None:
                trajectories.append(
                    {
                        "data": data,
                        "full_assistant_response": assistant_response,
                        "feedback": eval_result.feedback,
                    }
                )

        objective_scores_arg = None
        if objective_scores:
            all_none = all(x is None for x in objective_scores)
            all_not_none = all(x is not None for x in objective_scores)
            if not (all_none or all_not_none):
                raise ValueError(
                    "Objective scores must either be all None or all not None."
                )
            if all_not_none:
                objective_scores_arg = cast(list[dict[str, float]], objective_scores)

        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories,
            objective_scores=objective_scores_arg,
        )

    DefaultAdapter.evaluate = safe_evaluate


def main():
    load_env()
    patch_default_adapter()

    import gepa

    MODEL = "openai/gpt-5.4"
    REFLECTION_MODEL = "openai/gpt-5.4"
    MAX_WORKERS = 20

    trainset, valset, testset = load_aime2025()

    seed_candidate = {
        "system_prompt": (
            "You are a competition math solver. "
            "Solve the problem step by step. "
            "End your answer with '### <answer>'."
        )
    }

    print(f">>> model: {MODEL}")
    print(f">>> reflection: {REFLECTION_MODEL}")
    print(f">>> max_workers: {MAX_WORKERS}")
    print(f">>> train={len(trainset)}, val={len(valset)}, test={len(testset)}")
    print(f">>> seed: {seed_candidate['system_prompt']!r}\n")

    from gepa.adapters.default_adapter.default_adapter import DefaultAdapter

    adapter = DefaultAdapter(
        model=MODEL,
        max_litellm_workers=MAX_WORKERS,
    )

    result = gepa.optimize(
        seed_candidate=seed_candidate,
        trainset=trainset,
        valset=valset,
        adapter=adapter,
        reflection_lm=REFLECTION_MODEL,
        max_metric_calls=1000,
        reflection_minibatch_size=8,
        display_progress_bar=True,
        seed=0,
    )

    best_score = result.val_aggregate_scores[result.best_idx]
    print("\n" + "=" * 70)
    print(
        f"BEST VAL SCORE: {best_score:.3f} ({best_score * len(valset):.0f}/{len(valset)})"
    )
    print(f"BEST IDX: {result.best_idx}")
    print("=" * 70)
    print("BEST PROMPT:\n")
    print(result.best_candidate["system_prompt"])


if __name__ == "__main__":
    main()
