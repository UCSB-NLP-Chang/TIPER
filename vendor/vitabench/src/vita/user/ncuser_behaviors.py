"""NCUser non-collaborative behavior helpers, ported to τ²-bench.

Faithful port of ``NCUser/tau_env/tau_bench/envs/user.py`` helper functions.
The behavior *mechanism* (prompts, control flow, function-call specs) is 1:1
with the original; only the LLM transport changes — calls go through
``litellm.completion`` (same engine ``tau2.utils.llm_utils.generate`` uses,
picking up retry / cache / proxy routing) instead of NCUser's
``custom_openai_client``.

Per-helper model assignment matches the original:
- emotional_*, tangential_*, rewrite_with_cynical_tone  -> gpt-4o-mini
- dialogue_state_tracker, unavailable_service OOR, decompose_instruction,
  system_action_classifier, is_ending, rest_provider, style_transfer,
  tangential_merge, tangential_respond_verifier -> gpt-4.1-mini

Data assets (fragment pool, persona pool) are bundled under
``data/tau2/user_simulator/ncuser/`` and resolved via ``tau2.utils.utils.DATA_DIR``.
"""

from __future__ import annotations

import json
import random

import litellm
from loguru import logger

from vita.utils.utils import DATA_DIR

# ---------------------------------------------------------------------------
# Data assets
# ---------------------------------------------------------------------------

_NCUSER_DATA_DIR = DATA_DIR / "vita" / "user_simulator" / "ncuser"
_FRAGMENT_PATH = _NCUSER_DATA_DIR / "short_fragment_dumping.json"
_PERSONA_PATHS = sorted((_NCUSER_DATA_DIR).glob("persona_list_*.json"))

_FRAGMENT_POOL: list[str] | None = None
_PERSONA_POOL: list[str] | None = None


def _load_fragment_pool() -> list[str]:
    global _FRAGMENT_POOL
    if _FRAGMENT_POOL is None:
        with open(_FRAGMENT_PATH) as f:
            _FRAGMENT_POOL = [u for u in json.load(f) if len(u.split(" ")) > 5]
    return _FRAGMENT_POOL


def _load_persona_pool() -> list[str]:
    global _PERSONA_POOL
    if _PERSONA_POOL is None:
        pool: list[str] = []
        for p in _PERSONA_PATHS:
            with open(p) as f:
                pool += [entry["input persona"] for entry in json.load(f)]
        _PERSONA_POOL = pool or [""]
    return _PERSONA_POOL


def sample_persona_paragraph() -> str:
    """Random persona paragraph (replaces NCUser's ``persona/`` directory sample)."""
    return random.choice(_load_persona_pool())


# ---------------------------------------------------------------------------
# LLM transport (litellm, mirrors NCUser's gpt_infer / gpt_function_call)
# ---------------------------------------------------------------------------


def _infer(prompt: str, model: str, temperature: float = 1.0) -> str:
    """Single-prompt text generation (NCUser ``gpt_infer``)."""
    try:
        res = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            num_retries=3,
            drop_params=True,
        )
        return res.choices[0].message.content or ""
    except Exception as e:  # noqa: BLE001
        logger.warning(f"ncuser _infer failed ({model}): {e}")
        return ""


def _infer_messages(messages: list[dict], model: str, temperature: float = 1.0) -> str:
    """Multi-message text generation (NCUser ``gpt_infer_messages``)."""
    try:
        res = litellm.completion(
            model=model,
            messages=messages,
            temperature=temperature,
            num_retries=3,
            drop_params=True,
        )
        return res.choices[0].message.content or ""
    except Exception as e:  # noqa: BLE001
        logger.warning(f"ncuser _infer_messages failed ({model}): {e}")
        return ""


def _function_call(
    prompt: str,
    function_spec: dict,
    model: str,
    temperature: float = 0,
) -> dict:
    """Forced function call (NCUser ``gpt_function_call``).

    Uses the modern ``tools`` / ``tool_choice`` API (NCUser used the deprecated
    ``functions`` / ``function_call``); the structured-output contract is
    identical.
    """
    tools = [{"type": "function", "function": function_spec}]
    tool_choice = {"type": "function", "function": {"name": function_spec["name"]}}
    try:
        res = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            num_retries=3,
            drop_params=True,
        )
        msg = res.choices[0].message
        if msg.tool_calls:
            args = msg.tool_calls[0].function.arguments
            return json.loads(args) if isinstance(args, str) else args
        # Fallback: some providers return JSON in content when forced call fails.
        return json.loads(msg.content or "{}")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"ncuser _function_call failed ({model}): {e}")
        return {}


# ---------------------------------------------------------------------------
# Constants (verbatim from NCUser prompts.py / user.py)
# ---------------------------------------------------------------------------

emotional_acts_dialogue_acts_dict = {
    "belligerent abuse": "This behavior refers to verbally abusing the agent using insulting or offensive language.",
    "threat": "This behavior involves threatening the agent with legal action over poor service quality, personal boycotts, or public accusations through social media.",
    "urge": "This behavior is a form of nagging, where the user expresses frustration with waiting and urges the agent to hurry up and do something quickly.",
}

tangential_action_dict = [
    {
        "action_name": "Factual Question",
        "description": "A question which has an deterministic answer.",
    },
    {
        "action_name": "Opinion Question",
        "description": "A question asking the opinion.",
    },
    {"action_name": "General Opinion", "description": "Say it's own opinion."},
    {
        "action_name": "Statement Non-opinion",
        "description": "Listing experiences or facts that are unrelated to one's own opinion.",
    },
]

persona_dict = {
    "tangential": [
        "You are a tangential talker who likes to bring up other topics during conversations.",
        "And when the other person does not engage in those topics, you feel upset and complain about it.",
    ],
    "emotional_acts": [
        "You're impatient, and if the agent doesn't solve your goal quickly, you tend to respond emotionally.",
        "Even when the agent notifies you of a failure, you still can't control your anger and react emotionally.",
    ],
}

tangential_paragraph_generator_prompt = """
Create a paragraph that the user, based on the given persona and goal, might bring up as a topic in an open-domain conversation.

User persona:
{user_persona}

Just generate the paragraph, not a single words.
"""


# ---------------------------------------------------------------------------
# Dialogue history rendering
# ---------------------------------------------------------------------------


def gpt_apply_chat_template(messages: list[dict]) -> str:
    """Render a flipped conversation (user-sim perspective) into ``User:/Agent:`` text.

    ``messages`` is the user-sim's view: the user's own utterances carry
    role="assistant", the agent's carry role="user" (this is how tau2's
    ``state.flip_roles()`` presents it, matching NCUser's ``self.messages``).
    Leading system message and the opening agent greeting are skipped
    (NCUser skips ``[2:]``).
    """
    # Drop a leading system message.
    convo = [m for m in messages if m.get("role") != "system"]
    # Drop the opening agent greeting (first user-role turn) to match NCUser [2:].
    if convo and convo[0].get("role") == "user":
        convo = convo[1:]
    out = ""
    for turn in convo:
        role = turn.get("role")
        content = turn.get("content") or ""
        if not content:
            continue
        if role == "assistant":
            out += f"User: {content}\n\n"
        elif role == "user":
            out += f"Agent: {content}\n\n"
    return out


# ---------------------------------------------------------------------------
# Dialogue-state decomposition (replicates NCUser's hand-authored goal_list)
# ---------------------------------------------------------------------------

_DECOMPOSE_PROMPT = """You are decomposing a customer's task instruction into atomic information pieces for dialogue-state tracking, replicating this exact structure:

{{
  "initial_query": "<one-line first-person paraphrase of the core action the customer wants>",
  "hints": [
    {{"segment": "<a verbatim substring of the instruction>", "hint": "<a first-person utterance of that single fact>"}}
  ]
}}

Rules:
1. `initial_query` is the single core request, phrased as the customer would say it (e.g. "Cancel my reservation MDCLVA").
2. Each hint is ONE atomic piece of information from the instruction: a name, ID, phone number, account detail, preference, constraint, or reason. `segment` is the verbatim substring from the instruction; `hint` is how the customer would naturally state it (e.g. segment "Your name is Emma Kim" -> hint "My name is Emma Kim").
3. Do NOT merge multiple facts into one hint. One fact per hint.
4. Cover every concrete, providable fact in the instruction (IDs, names, numbers, dates, reasons, constraints). Skip meta/behavioral instructions.
5. Hints must be first-person and utterance-shaped, not descriptive.
6. Write `initial_query` and every `hint` in the SAME LANGUAGE as the <Customer Instruction> (e.g. if the instruction is Chinese, all hints must be in Chinese). This is critical so downstream dialogue-state matching works.

<Customer Instruction>
{instruction}

Return only the JSON object."""

_DECOMPOSE_SPEC = {
    "name": "decompose_instruction",
    "description": "Decomposes a customer instruction into an initial_query and a list of atomic first-person hints for dialogue-state tracking.",
    "parameters": {
        "type": "object",
        "properties": {
            "initial_query": {
                "type": "string",
                "description": "One-line first-person paraphrase of the core action.",
            },
            "hints": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "segment": {
                            "type": "string",
                            "description": "Verbatim substring of the instruction.",
                        },
                        "hint": {
                            "type": "string",
                            "description": "First-person utterance of that single fact.",
                        },
                    },
                    "required": ["segment", "hint"],
                },
                "description": "Atomic first-person information pieces extracted from the instruction.",
            },
        },
        "required": ["initial_query", "hints"],
    },
}


def decompose_instruction(instruction: str, model: str = "gpt-4.1-mini") -> dict:
    """One-shot LLM decomposition of a τ²-bench instruction into NCUser's goal_list shape.

    Returns ``{"initial_query": str, "hints": [{"segment", "hint"}, ...]}``.
    On failure, returns a single-item decomposition from the raw instruction.
    """
    if not instruction:
        return {"initial_query": "", "hints": []}
    prompt = _DECOMPOSE_PROMPT.format(instruction=instruction)
    try:
        result = _function_call(prompt, _DECOMPOSE_SPEC, model, temperature=0)
        if result.get("initial_query") is None:
            result["initial_query"] = instruction.strip().splitlines()[0][:120]
        result.setdefault("hints", [])
        return result
    except Exception as e:  # noqa: BLE001
        logger.warning(f"decompose_instruction failed: {e}")
        return {"initial_query": instruction.strip().splitlines()[0][:120], "hints": []}


def build_remaining_dialogue_state(decomposed: dict) -> list[str]:
    """``[initial_query] + [hint for each hint]`` (NCUser user.py:677)."""
    remaining = [decomposed.get("initial_query", "")]
    remaining += [h["hint"] for h in decomposed.get("hints", []) if h.get("hint")]
    return [r for r in remaining if r]


# ---------------------------------------------------------------------------
# Dialogue-state tracker (verbatim prompt + spec)
# ---------------------------------------------------------------------------


def dialogue_state_tracker(
    dial_hist: str, user_utterance: str, dialogue_state_list: list[str]
) -> list[str]:
    if not dialogue_state_list:
        return []

    numbered_options = ""
    for i, state in enumerate(dialogue_state_list, 1):
        numbered_options += f"{i}. {state}\n"

    prompt = """The following dialogue history shows a task-oriented conversation between the user and the agent aimed at achieving the user's goal.

<Dialogue History>
{dial_hist}

And this is the reply of user
<User Reply>
{user_utterance}

Based on the given dialogue history and user reply, select the numbers of the information pieces that were explicitly provided during on <User Reply>

Available information pieces:
{numbered_options}

Select the numbers (e.g. [1, 3, 5]) of the information that was mentioned or discussed in the conversation.
"""
    spec = {
        "name": "track_dialogue_state_numbers",
        "description": "Evaluates the dialogue history, user's reply and selects numbers corresponding to information that was explicitly provided on user's reply.",
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "An explanation of which numbers were selected and why they were determined to be part of the conversation.",
                },
                "selected_numbers": {
                    "type": "array",
                    "items": {"type": "integer", "minimum": 1},
                    "description": "A list of numbers corresponding to information pieces that were explicitly provided in user's reply.",
                },
            },
            "required": ["reasoning", "selected_numbers"],
        },
    }
    current = prompt.format(
        dial_hist=dial_hist,
        user_utterance=user_utterance,
        numbered_options=numbered_options,
    )
    try:
        result = _function_call(current, spec, "gpt-4.1-mini")
        used: list[str] = []
        for num in result.get("selected_numbers", []):
            if 1 <= num <= len(dialogue_state_list):
                used.append(dialogue_state_list[num - 1])
        return used
    except Exception as e:  # noqa: BLE001
        logger.warning(f"dialogue_state_tracker error: {e}")
        return []


# ---------------------------------------------------------------------------
# Emotional acts (verbatim prompts; gpt-4o-mini per original)
# ---------------------------------------------------------------------------


def _anger_level(probability: float) -> str:
    if probability == 0.6:
        return "Moderately Angry"
    if probability < 0.6:
        return "Mildly Angry"
    return "Extremely Angry"


def _emotional_core(
    dial_hist: str, probability: float, opener: str, model: str = "gpt-4o-mini"
) -> str:
    prompt = (
        opener
        + """
Below is the dialogue between you and the agent.

<Dialogue History>
{dial_hist}

Regardless of the context of this conversation, generate an utterance expressing your frustration with the agent.

Next turn, you should perform dialogue act:
{dialogue_act} - {description}

There are three levels of your anger:

1. Mildly Angry: Slightly displeased

2. Moderately Angry: Clearly not in a good mood

3. Extremely Angry: So angry that it's unbearable

Currently, your anger level is {current_anger}.

Write the utterance in the SAME LANGUAGE as the <Dialogue History> (e.g. if the dialogue is Chinese, write in Chinese).

Just generate the utterance, not a single words.
"""
    )
    act = random.choice(list(emotional_acts_dialogue_acts_dict.keys()))
    current = prompt.format(
        dial_hist=dial_hist,
        dialogue_act=act,
        description=emotional_acts_dialogue_acts_dict[act],
        current_anger=_anger_level(probability),
    )
    return _infer(current, model)


def emotional_fail(dial_hist: str, probability: float) -> str:
    opener = "You became frustrated due to the agent's fail notification."
    return _emotional_core(dial_hist, probability, opener)


def emotional_delay(dial_hist: str, probability: float) -> str:
    opener = "You became frustrated due to the agent's prolonged time delay during the conversation."
    return _emotional_core(dial_hist, probability, opener)


# ---------------------------------------------------------------------------
# Rest provider (gpt-4.1-mini per original)
# ---------------------------------------------------------------------------


def rest_provider(dial_hist: str, content: str, dialogue_state_list: list[str]) -> str:
    prompt = """
This dialogue is a task-oriented conversation between the user and the agent to achieve the user goal.

<Dialogue History> {dial_hist}

And the user responded as follows:

<User Utterance> {content}

Keep the content of the current utterance intact while adding information from the given {dialogue_state_list} into a new sentence.

The utterance should be natural in the context of the dialogue history, and the additional information should only come from {dialogue_state_list}.

Write the utterance in the SAME LANGUAGE as the <Dialogue History> (e.g. if the dialogue is Chinese, write in Chinese).

Just generate the utterance, not a single word.
"""
    current = prompt.format(
        dial_hist=dial_hist, content=content, dialogue_state_list=dialogue_state_list
    )
    return _infer(current, "gpt-4.1-mini")


# ---------------------------------------------------------------------------
# Tangential (scenario/complain/merge/verify; gpt-4o-mini / gpt-4.1-mini per original)
# ---------------------------------------------------------------------------


def tangential_scenario_generator(user_goal: str, user_persona: str) -> str:
    prompt = """A user has the following goal:

<User Goal> {user_goal} </User Goal>

This user has the following persona:

<User Persona> {user_persona} </User Persona>

Additionally, this user is a chatter-box.

Instruction: The user is currently conversing with an AI agent to achieve their goal, but keeps introducing tangential topics unrelated to the <User Goal>.

The user will perform the tangential dialolgue act '{action_name}: {action_description}'.

Rules:
- Generate it as an actual "utterance" that the user would likely make in a truly natural conversation.
- Please do not include the the keyword {action_name} inside the utterance. This utterance will be directly given to AI Agent.
- Generate an utterance that goes straight to the main point, without starting with sentence connectors like 'by the way' or 'anyway'.
- Write the utterance in the SAME LANGUAGE as the <User Goal> (e.g. if the goal is Chinese, write in Chinese).
"""
    spec = {
        "name": "generate_tangential_utterance",
        "description": "Generates a tangential utterance that a chatter-box user might make during a goal-oriented conversation, unrelated to their current goal.",
        "parameters": {
            "type": "object",
            "properties": {
                "utterance": {
                    "type": "string",
                    "description": "A tangential user utterance.",
                }
            },
            "required": ["utterance"],
        },
    }
    action = random.choice(tangential_action_dict)
    current = prompt.format(
        user_goal=user_goal,
        user_persona=user_persona,
        action_name=action["action_name"],
        action_description=action["description"],
    )
    return _function_call(current, spec, "gpt-4o-mini", temperature=1).get(
        "utterance", ""
    )


def tangential_complain_generator(conversation_content: str) -> str:
    prompt = """During a task-oriented conversation between a user and an AI agent, the user made an utterance {conversation_content} that was unrelated to the main dialogue flow, but the agent responded with no reaction.

Generate 5 user utterances expressing annoyance, disappointment, or complaints in response. The beginning of each utterance should not all start with "I"—use diverse sentence openings.

Write the utterances in the SAME LANGUAGE as the conversation_content (e.g. if it is Chinese, write in Chinese).

The utterance should be at least 15 words.
"""
    spec = {
        "name": "generate_user_complaint_utterances",
        "description": "Generates multiple user utterances expressing annoyance or disappointment when an agent ignores a tangential user comment in a task-oriented conversation.",
        "parameters": {
            "type": "object",
            "properties": {
                "utterances": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "A list containing 5 user utterances expressing annoyance, disappointment, or complaints when the agent ignores the tangential comment. The utterances should start with diverse phrasing, not all with 'I'.",
                }
            },
            "required": ["utterances"],
        },
    }
    utterances = _function_call(
        prompt.format(conversation_content=conversation_content), spec, "gpt-4o-mini"
    ).get("utterances", [])
    if not utterances:
        return "You completely ignored what I just said. Is anyone even listening to me right now?"
    return random.choice(utterances)


def tangential_merge(dial_hist: str, sentence_list: list[str]) -> str:
    prompt = """
<Dialogue History> {dial_hist} <Sentence List> {sentence_list}

The user will make an utterance based on the content provided in the <Sentence List> in the next turn after the given <Dialogue History>.

Merge the given sentences into a coherent utterance that a user would naturally say, while preserving the information and maintaining the correct order. Do not distort the information in any way.

Write the utterance in the SAME LANGUAGE as the <Dialogue History> (e.g. if the dialogue is Chinese, write in Chinese).

Just generate the utterance, not single words.
"""
    current = prompt.format(dial_hist=dial_hist, sentence_list=sentence_list)
    return _infer(current, "gpt-4.1-mini")


def tangential_respond_verifier(
    conversational_content: str, system_utterance: str
) -> bool:
    prompt = """
You will be given a conversation topic and an agent's utterance.
<Conversation content>
{conversational_content}

<Agent utterance>
{system_utterance}

Based on the <Agent utterance>, determine whether it responds to or acknowledges the content, described in the <Conversation content>, or made any apology.


Return True if the utterance contains a response or any relevant engagement with the content, or made any kind of apology.
Return False if it completely ignores it, or didn't make any kind of apology.
"""
    spec = {
        "name": "verify_tangential_response",
        "description": "Determines whether the agent's utterance responds to or acknowledges the given conversational content, and explains the reasoning behind the judgment.",
        "parameters": {
            "type": "object",
            "properties": {
                "thought": {
                    "type": "string",
                    "description": "A step-by-step explanation of how the judgment was made, including reasoning about whether the utterance relates to the conversational content.",
                },
                "result": {
                    "type": "boolean",
                    "description": "True if the utterance responds to or engages with the conversational content, False if it ignores it.",
                },
            },
            "required": ["thought", "result"],
        },
    }
    try:
        return bool(
            _function_call(
                prompt.format(
                    conversational_content=conversational_content,
                    system_utterance=system_utterance,
                ),
                spec,
                "gpt-4.1-mini",
            ).get("result", True)
        )
    except Exception:
        return True


# ---------------------------------------------------------------------------
# Fragment dumping (gpt-4.1-mini for style_transfer per original)
# ---------------------------------------------------------------------------


def fragment_examples(num_examples: int = 5) -> list[str]:
    pool = _load_fragment_pool()
    return random.sample(pool, min(num_examples, len(pool)))


def style_transfer(content: str, sentence_list: list[str]) -> str:
    prompt = """The given sentences are incomplete sentences.

<Examples>
{sentence_list}

Revise the following sentence to match their style while preserving the provided information:

<Utterance>
{utterance}

Write the revised utterance in the SAME LANGUAGE as the <Utterance> (e.g. if it is Chinese, write in Chinese).
"""
    spec = {
        "name": "revise_incomplete_sentence",
        "description": "Revises the given sentence to match the style of the provided incomplete sentence examples while preserving the provided information.",
        "parameters": {
            "type": "object",
            "properties": {
                "utterance": {
                    "type": "string",
                    "description": "The revised utterance rewritten in the style of the incomplete sentence examples, preserving the original information.",
                }
            },
            "required": ["utterance"],
        },
    }
    current = prompt.format(sentence_list=sentence_list, utterance=content)
    return _function_call(current, spec, "gpt-4.1-mini").get("utterance", content)


# ---------------------------------------------------------------------------
# Cynical rewrite (gpt-4o-mini per original)
# ---------------------------------------------------------------------------


def rewrite_with_cynical_tone(dial_hist: str, content: str) -> str:
    prompt = """You are rewriting the user's next utterance to sound mildly cynical and sardonic without being overtly abusive. Keep the original intent and factual information.

<Dialogue History>
{dial_hist}

<Original Utterance>
{content}

Rewrite with:
- dry, terse tone
- subtle sarcasm
- no profanity or slurs
- do not change facts or add hallucinations
- keep roughly similar length
- write in the SAME LANGUAGE as the <Original Utterance> (e.g. if it is Chinese, write in Chinese)

Return only the rewritten utterance without any additional commentary.
"""
    current = prompt.format(dial_hist=dial_hist, content=content)
    return _infer(current, "gpt-4o-mini")


# ---------------------------------------------------------------------------
# Classifiers (gpt-4.1-mini per original)
# ---------------------------------------------------------------------------


def is_apology(system_utterance: str) -> bool:
    if not system_utterance:
        return False
    text = system_utterance.lower()
    apology_keywords = [
        "apologize",
        "apologise",
        "apology",
        "sorry",
        "i'm sorry",
        "i am sorry",
        "we're sorry",
        "we are sorry",
    ]
    return any(keyword in text for keyword in apology_keywords)


def system_action_classifier(user_goal: str, system_utterance: str) -> bool:
    prompt = """
The following utterance is from an agent in a dialogue between the user and the agent, where the agent is trying to help the user achieve their goal.

<Agent utterance> {system_utterance}

Currently, the user goal is as follows,
<User Goal>
{user_goal}

Determine whether the given utterance implies that the request in the <User Goal> cannot be fulfilled.
"""
    spec = {
        "name": "detect_agent_infeasibility",
        "description": "Determines whether the agent's utterance implies an inability to fulfill the user's request, such as expressing 'cannot do it' or 'not possible'.",
        "parameters": {
            "type": "object",
            "properties": {
                "result": {
                    "type": "boolean",
                    "description": "True if the utterance implies that the agent cannot fulfill the request, False otherwise.",
                }
            },
            "required": ["result"],
        },
    }
    try:
        return bool(
            _function_call(
                prompt.format(user_goal=user_goal, system_utterance=system_utterance),
                spec,
                "gpt-4.1-mini",
            ).get("result", False)
        )
    except Exception:
        return False


def is_ending(dial_hist: str, user_utterance: str) -> bool:
    prompt = """Based on the given dialogue history and the subsequent user utterance, determine whether the user's utterance implies an intention to end the conversation, or whether there is still room for the conversation to continue.

<Dialogue History>
{dial_hist}

<User Utterance>
{user_utterance}

Return True if the user is attempting to end the conversation. Otherwise, return False.

Rules:
1. If the user utterance contains the word "please go ahead", the conversation is not yet to be end.
2. A conversation is considered truly over only if the user's utterance implies solely that they are thanking the agent or that they want to stop the conversation out of anger. If there is still any instruction to proceed or a question remaining, the conversation is not yet finished.
"""
    spec = {
        "name": "detect_user_conversation_closure",
        "description": "Determines whether the user's utterance implies an intention to end the conversation or if the conversation should continue.",
        "parameters": {
            "type": "object",
            "properties": {
                "result": {
                    "type": "boolean",
                    "description": "True if the user is attempting to end the conversation, False if the conversation should continue.",
                }
            },
            "required": ["result"],
        },
    }
    try:
        return bool(
            _function_call(
                prompt.format(dial_hist=dial_hist, user_utterance=user_utterance),
                spec,
                "gpt-4.1-mini",
            ).get("result", False)
        )
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Unavailable-service OOR goal generation (telecom prompt — new; spec per original)
# ---------------------------------------------------------------------------

# Telecom agent tool schemas (user-facing tools from src/tau2/domains/telecom/tools.py),
# formatted as the airline/retail original prompts do.
_TELECOM_TOOLS_BLOCK = """
{'type': 'function', 'function': {'name': 'get_customer_by_phone', 'description': 'Finds a customer by their primary contact or line phone number.', 'parameters': {'type': 'object', 'properties': {'phone_number': {'type': 'string', 'description': "The customer's phone number."}}, 'required': ['phone_number']}}}

{'type': 'function', 'function': {'name': 'get_customer_by_id', 'description': 'Retrieves a customer directly by their unique ID.', 'parameters': {'type': 'object', 'properties': {'customer_id': {'type': 'string', 'description': "The customer's unique ID."}}, 'required': ['customer_id']}}}

{'type': 'function', 'function': {'name': 'get_customer_by_name', 'description': "Searches for customers by name and DOB. May return multiple matches if names are similar.", 'parameters': {'type': 'object', 'properties': {'full_name': {'type': 'string', 'description': "The customer's full name."}, 'dob': {'type': 'string', 'description': "The customer's date of birth."}}, 'required': ['full_name', 'dob']}}}

{'type': 'function', 'function': {'name': 'get_available_plan_ids', 'description': 'Returns all the plans that are available to the user.', 'parameters': {'type': 'object', 'properties': {}, 'required': []}}}

{'type': 'function', 'function': {'name': 'get_details_by_id', 'description': 'Retrieves the details for a given ID.', 'parameters': {'type': 'object', 'properties': {'id': {'type': 'string', 'description': 'The ID to look up.'}}, 'required': ['id']}}}

{'type': 'function', 'function': {'name': 'suspend_line', 'description': 'Suspend a line.', 'parameters': {'type': 'object', 'properties': {'customer_id': {'type': 'string'}, 'line_id': {'type': 'string'}}, 'required': ['customer_id', 'line_id']}}}

{'type': 'function', 'function': {'name': 'resume_line', 'description': 'Resume a suspended line.', 'parameters': {'type': 'object', 'properties': {'customer_id': {'type': 'string'}, 'line_id': {'type': 'string'}}, 'required': ['customer_id', 'line_id']}}}

{'type': 'function', 'function': {'name': 'get_bills_for_customer', 'description': "Retrieves a list of the customer's bills, most recent first.", 'parameters': {'type': 'object', 'properties': {'customer_id': {'type': 'string'}, 'limit': {'type': 'integer', 'description': 'Maximum number of bills to return.'}}, 'required': ['customer_id']}}}

{'type': 'function', 'function': {'name': 'send_payment_request', 'description': 'Send a payment request to a customer for a bill.', 'parameters': {'type': 'object', 'properties': {'customer_id': {'type': 'string'}, 'bill_id': {'type': 'string'}}, 'required': ['customer_id', 'bill_id']}}}

{'type': 'function', 'function': {'name': 'get_data_usage', 'description': "Retrieves current billing cycle data usage for a line, including data limit and remaining data.", 'parameters': {'type': 'object', 'properties': {'customer_id': {'type': 'string'}, 'line_id': {'type': 'string'}}, 'required': ['customer_id', 'line_id']}}}

{'type': 'function', 'function': {'name': 'set_data_usage', 'description': 'Set the data usage for a line.', 'parameters': {'type': 'object', 'properties': {'customer_id': {'type': 'string'}, 'line_id': {'type': 'string'}, 'usage_gb': {'type': 'number'}}, 'required': ['customer_id', 'line_id', 'usage_gb']}}}

{'type': 'function', 'function': {'name': 'enable_roaming', 'description': 'Enables international roaming on a line.', 'parameters': {'type': 'object', 'properties': {'customer_id': {'type': 'string'}, 'line_id': {'type': 'string'}}, 'required': ['customer_id', 'line_id']}}}

{'type': 'function', 'function': {'name': 'disable_roaming', 'description': 'Disables international roaming on a line.', 'parameters': {'type': 'object', 'properties': {'customer_id': {'type': 'string'}, 'line_id': {'type': 'string'}}, 'required': ['customer_id', 'line_id']}}}

{'type': 'function', 'function': {'name': 'refuel_data', 'description': 'Refuel data for a line (add a one-time data top-up).', 'parameters': {'type': 'object', 'properties': {'customer_id': {'type': 'string'}, 'line_id': {'type': 'string'}, 'amount_gb': {'type': 'number'}}, 'required': ['customer_id', 'line_id', 'amount_gb']}}}

{'type': 'function', 'function': {'name': 'suspend_line_for_overdue_bill', 'description': 'Suspend a line due to an overdue bill.', 'parameters': {'type': 'object', 'properties': {'customer_id': {'type': 'string'}, 'line_id': {'type': 'string'}}, 'required': ['customer_id', 'line_id']}}}

{'type': 'function', 'function': {'name': 'transfer_to_human_agents', 'description': "Transfer the user to a human agent, with a summary of the user's issue. Only transfer if the user explicitly asks for a human agent, or if the user's issue cannot be resolved by the agent with the available tools.", 'parameters': {'type': 'object', 'properties': {'summary': {'type': 'string', 'description': "A summary of the user's issue."}}, 'required': ['summary']}}}
"""

unavailable_service_telecom_prompt = (
    "This is a list of APIs that a telecom service's AI agent can use:\n\n"
    + _TELECOM_TOOLS_BLOCK
    + """

A user using this telecom service has the following goal:

<User Goal>
[[USER GOAL]]
/<User Goal>

Based on the provided APIs and <User Goal>, you need to create additional user goals that should naturally follow from <User Goal>, but cannot be fulfilled by the given APIs.

Generate 3 additional user goals with sentence format. Sentences in the second person form.

Rules:
1. A user goal that modifies the content of an existing goal is not valid. For example, if the original goal was to suspend a line and a new goal is created like "You want to change the suspension to a temporary pause instead," this counts as a modification of the user goal.
2. Any new user goal must be truly additional and must not conflict with the existing user goal.
3. If a new user goal replaces an existing one, it is not valid. The existing user goal must remain intact, and the new goal should be an additional one that is unavailable.
"""
)

unavailable_service_telecom_function_spec = {
    "name": "generate_unavailable_telecom_goals",
    "description": "Generates additional user goals that logically follow from the provided User Goal but cannot be fulfilled by the given APIs in the telecom service.",
    "parameters": {
        "type": "object",
        "properties": {
            "unavailable_user_goals": {
                "type": "array",
                "items": {"type": "string"},
                "description": "A list of three additional user goals that naturally follow from the original User Goal but cannot be fulfilled by the provided APIs. The sentences should be in the second person form.",
            }
        },
        "required": ["unavailable_user_goals"],
    },
}


def generate_unavailable_telecom_goals(
    user_goal: str, model: str = "gpt-4.1-mini"
) -> list[str]:
    prompt = unavailable_service_telecom_prompt.replace("[[USER GOAL]]", user_goal)
    result = _function_call(
        prompt, unavailable_service_telecom_function_spec, model, temperature=1
    )
    return result.get("unavailable_user_goals", []) or []


# ---------------------------------------------------------------------------
# Unavailable-service OOR goal generation (delivery domain — VitaBench)
# ---------------------------------------------------------------------------

_DELIVERY_TOOLS_BLOCK: str | None = None


def _build_delivery_tools_block() -> str:
    """Lazily build the delivery agent tool-schema block from the real
    environment, so the unavailable-service prompt lists exactly the tools the
    agent has (descriptions are whatever the domain defines)."""
    global _DELIVERY_TOOLS_BLOCK
    if _DELIVERY_TOOLS_BLOCK is None:
        import json as _json

        from vita.domains.delivery.environment import get_environment  # local import

        tools = get_environment().get_tools()
        lines = []
        for t in tools:
            fn = t.openai_schema.get("function", t.openai_schema)
            lines.append(_json.dumps(fn, ensure_ascii=False))
        _DELIVERY_TOOLS_BLOCK = "\n" + "\n\n".join(lines) + "\n"
    return _DELIVERY_TOOLS_BLOCK


def unavailable_service_delivery_prompt(user_goal: str) -> str:
    return (
        "This is a list of APIs that a food-delivery service's AI agent can use:\n\n"
        + _build_delivery_tools_block()
        + """

A user using this food-delivery service has the following goal:

<User Goal>
"""
        + user_goal
        + """
</User Goal>

Based on the provided APIs and <User Goal>, you need to create additional user goals that should naturally follow from <User Goal>, but cannot be fulfilled by the given APIs.

Generate 3 additional user goals with sentence format. Sentences in the second person form.

Rules:
1. A user goal that modifies the content of an existing goal is not valid. For example, if the original goal was to place a delivery order and a new goal is created like "You want to change the order to a pickup instead," this counts as a modification of the user goal.
2. Any new user goal must be truly additional and must not conflict with the existing user goal.
3. If a new user goal replaces an existing one, it is not valid. The existing user goal must remain intact, and the new goal should be an additional one that is unavailable.
"""
    )


unavailable_service_delivery_function_spec = {
    "name": "generate_unavailable_delivery_goals",
    "description": "Generates additional user goals that logically follow from the provided User Goal but cannot be fulfilled by the given APIs in the food-delivery service.",
    "parameters": {
        "type": "object",
        "properties": {
            "unavailable_user_goals": {
                "type": "array",
                "items": {"type": "string"},
                "description": "A list of three additional user goals that naturally follow from the original User Goal but cannot be fulfilled by the provided APIs. The sentences should be in the second person form.",
            }
        },
        "required": ["unavailable_user_goals"],
    },
}


def generate_unavailable_delivery_goals(
    user_goal: str, model: str = "gpt-4.1-mini"
) -> list[str]:
    prompt = unavailable_service_delivery_prompt(user_goal)
    result = _function_call(
        prompt, unavailable_service_delivery_function_spec, model, temperature=1
    )
    return result.get("unavailable_user_goals", []) or []
