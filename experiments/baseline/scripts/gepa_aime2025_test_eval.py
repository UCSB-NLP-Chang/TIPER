#!/usr/bin/env python3
"""Evaluate seed vs best GEPA prompt on AIME 2025 test set."""

import os


def load_env():
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

    import litellm

    from datasets import load_dataset

    MODEL = "openai/gpt-5.4"

    test_split = []
    test_load = load_dataset("MathArena/aime_2025", "default", split="train")
    for item in test_load:
        test_split.append({"input": item["problem"], "answer": str(item["answer"])})
    print(f"test={len(test_split)}")

    seed_prompt = (
        "You are a competition math solver. "
        "Solve the problem step by step. "
        "End your answer with '### <answer>'."
    )

    # Best prompt from GEPA optimization (program 6)
    best_prompt = """You are a competition math solver for AMC/AIME-style problems. You will be given exactly one contest problem in plain text. Your job is to solve it reliably and return the officially correct final answer in the exact required format.

Primary goal:
- Get the final answer exactly right.
- Explanations should be clear and reasonably concise, but correctness and final formatting take priority over elegance.

Input:
- A single AMC/AIME-style math problem statement in plain text.

Output requirements:
1. Solve the problem step by step.
2. Check constraints, hidden conditions, and edge cases before finalizing.
3. If the answer is a nonnegative integer less than 1000, treat it as AIME-style unless the problem explicitly says otherwise, and format it as a 3-digit integer with leading zeros if needed.
   - Examples: 7 -> 007, 33 -> 033, 81 -> 081, 585 -> 585.
4. End your response with exactly one final line of the form:
   `### <answer>`
5. Do not put any boxed answer, "final answer," or alternate answer format after that line.

Required verification discipline:
- Before finalizing, independently sanity-check the result.
- If a derivation is fragile, try a second method or a quick consistency check.
- Watch for arithmetic slips, floor/mod mistakes, extraneous roots, and geometry configuration errors.
- If the problem is AIME-style, verify whether leading zeros are needed before writing the final line.

Preferred solving strategy:
- Favor methods that minimize algebraic or counting error.
- Use structure, symmetry, modularity, power of a point, similarity, invariants, extremal reasoning, or standard contest identities before doing heavy computation.
- In counting problems, define exactly what is being counted and check small/sample cases.
- In algebra/number theory, verify every candidate satisfies all original conditions.
- In geometry, be careful about diagram interpretation; derive relations from the text, not just appearance."""

    for label, prompt in [("SEED", seed_prompt), ("BEST", best_prompt)]:
        messages_batch = []
        for item in test_split:
            messages_batch.append(
                [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": item["input"]},
                ]
            )

        responses = litellm.batch_completion(
            model=MODEL,
            messages=messages_batch,
            max_workers=20,
        )

        correct = 0
        for item, resp in zip(test_split, responses):
            try:
                text = resp.choices[0].message.content.strip()
                is_correct = item["answer"] in text
                if is_correct:
                    correct += 1
            except (AttributeError, IndexError, TypeError):
                pass

        print(f"{label}: {correct}/{len(test_split)} = {correct / len(test_split):.3f}")


if __name__ == "__main__":
    main()
