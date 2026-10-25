"""
OpenAI-powered reproduction of ColBench `code` tasks.

This is a drop-in alternative to `simulate_interactions.py` that replaces the
local vLLM policy (`VLLMAgent`) and the local vLLM human-simulator server with
direct OpenAI API calls, so it runs on a laptop without GPUs.

- The agent is `OpenAIAgent` (same `get_action(batch_obs)` interface as
  `VLLMAgent`), calling the OpenAI Chat Completions API.
- The human simulator reuses `HumanInteractionEnv`, whose `invoke_model` already
  speaks the OpenAI Chat Completions API — we just hand it a real OpenAI client
  pointed at `api.openai.com` instead of a local vLLM server.

Usage:
    export OPENAI_API_KEY=sk-...
    python scripts/simulate_interactions_openai.py \
        --input_path data/backend_tasks/test.jsonl \
        --output_path out_openai.jsonl \
        --num_tasks 10 --max_steps 10 \
        --agent_model gpt-4o-mini --human_model gpt-4o-mini
"""

import concurrent
import json
import os

from fire import Fire
from openai import OpenAI
from sweet_rl.environments.human_interaction_env import HumanInteractionEnv
from tqdm import tqdm


class OpenAIAgent:
    """Agent backed by the OpenAI Chat Completions API.

    Mirrors VLLMAgent.get_action: takes a list of dialogue histories (each a
    list of {"role","content"} dicts, or None for finished envs) and returns
    (formatted_prompts, responses) aligned with the input order.
    """

    def __init__(self, client, model_id, agent_prompt, temperature=0.7, max_tokens=1024):
        self.client = client
        self.model_id = model_id
        self.agent_prompt = agent_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _complete(self, messages):
        for _ in range(3):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                return completion.choices[0].message.content
            except Exception as e:  # noqa: BLE001 - retry transient API errors
                print(f"[OpenAIAgent] retrying after error: {e}")
        return "I WANT TO ANSWER:"

    def get_action(self, str_dialogue_histories, use_tqdm=True):
        # Map non-None histories to indices (finished envs return None).
        new_index2original = {}
        undone = []
        for i, history in enumerate(str_dialogue_histories):
            if history is not None:
                new_index2original[len(undone)] = i
                undone.append(history)

        if len(undone) == 0:
            return [None for _ in str_dialogue_histories], [
                None for _ in str_dialogue_histories
            ]

        def build_messages(history):
            return [{"role": "system", "content": self.agent_prompt}] + history

        formatted_prompts = [json.dumps(build_messages(h)) for h in undone]

        # Fire all completions concurrently; OpenAI client is thread-safe.
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            outputs = list(
                executor.map(
                    lambda m: self._complete(m), [build_messages(h) for h in undone]
                )
            )

        real_outputs = [None for _ in str_dialogue_histories]
        real_formatted = [None for _ in str_dialogue_histories]
        for j, out in enumerate(outputs):
            orig = new_index2original[j]
            real_outputs[orig] = out
            real_formatted[orig] = formatted_prompts[j]
        return real_formatted, real_outputs


def batch_interact_environment(agent, environments, tasks):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        jobs = [
            executor.submit(env.reset, task["problem_description"], task["ground_truth"])
            for env, task in zip(environments, tasks)
        ]
        batch_obs = [job.result() for job in jobs]
    for _ in range(environments[0].max_steps + 1):  # +1 for safety, matches upstream
        formatted_prompts, responses = agent.get_action(batch_obs)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            jobs = [
                executor.submit(env.step, response, formatted_prompt)
                for env, response, formatted_prompt in zip(
                    environments, responses, formatted_prompts
                )
            ]
            batch_obs = [job.result()[0] for job in jobs]
    return [
        {"task": task, "dialogue_history": env.dialogue_history, "answer": env.answer}
        for task, env in zip(tasks, environments)
    ]


def main(
    input_path=None,
    output_path="out_openai.jsonl",
    user_prompt_path="prompts/human_simulator_code_prompt.txt",
    agent_prompt_path="prompts/llm_agent_code_prompt.txt",
    agent_model="gpt-4o-mini",
    human_model="gpt-4o-mini",
    num_tasks=10,
    max_steps=10,
    temperature=0.7,
    evaluate=True,
):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY in the environment first.")
    client = OpenAI(api_key=api_key)

    with open(user_prompt_path, "r") as fb:
        human_prompt = fb.read()
    with open(agent_prompt_path, "r") as fb:
        agent_prompt = fb.read()

    with open(input_path, "r") as fb:
        tasks = [json.loads(line) for line in fb]
    tasks = tasks[:num_tasks]
    print(f"********************Number of tasks: {len(tasks)}**************")
    print(f"agent_model={agent_model}  human_model={human_model}  max_steps={max_steps}")

    environments = [
        HumanInteractionEnv(
            client=client,
            human_prompt=human_prompt,
            model_id=human_model,
            env_id=i,
            max_steps=max_steps,
        )
        for i in range(len(tasks))
    ]
    agent = OpenAIAgent(
        client=client,
        model_id=agent_model,
        agent_prompt=agent_prompt,
        temperature=temperature,
    )

    trajectory = batch_interact_environment(agent, environments, tasks)

    with open(output_path, "w") as fb:
        for d in trajectory:
            fb.write(json.dumps(d) + "\n")
    print(f"Wrote {len(trajectory)} trajectories to {output_path}")

    if evaluate:
        from sweet_rl.utils import code_evaluate

        rewards = code_evaluate(trajectory)
        for d, r in zip(trajectory, rewards):
            d["reward"] = r
        with open(output_path, "w") as fb:
            for d in trajectory:
                fb.write(json.dumps(d) + "\n")
        n_correct = sum(1 for r in rewards if r == 1)
        print(
            f"RESULT: {n_correct}/{len(rewards)} tasks fully correct "
            f"({n_correct / len(rewards) * 100:.1f}%), "
            f"avg test-case pass rate={sum(rewards) / len(rewards):.3f}"
        )


if __name__ == "__main__":
    Fire(main)
