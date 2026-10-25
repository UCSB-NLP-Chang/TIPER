#!/usr/bin/env python3
"""LLM majority-vote conversation quality filter.

This is the file-based version of the upstream `llm_quality_check.py` workflow.
It preprocesses simple artifacts, asks several model votes whether each
conversation is clean, and writes the clean subset for persona induction.

Example:
  export OPENAI_API_KEY=...
  .venv/bin/python -m cresta.persona.data_cleaning.llm_quality_filter \
    --input /path/to/raw_10k_conversations.json \
    --output-clean /path/to/persona-state/llm_clean_3k.json \
    --output-full /path/to/persona-state/llm_quality_results.json \
    --target-clean 3000
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

SSML_RE = re.compile(
    r"</?(?:break|speak|prosody|say-as|phoneme|sub|voice|audio)[^>]*/?>", re.I
)
DTMF_RE = re.compile(r"\[/?DTMF[^\]]*\]", re.I)
DEBUG_RE = re.compile(r"^\s*for debugging\b", re.I)

CATEGORIES = {
    "markup_artifact": (
        "Non-standard markup, tags, or encoding artifacts in message text "
        "(for example, '[[CREDIT-CARD]]' or '[Custom AI Agent]'). "
        "Standard redaction markers like [PHONE], [NAME], [EMAIL], and [ADDRESS] are not artifacts."
    ),
    "concatenated_turns": (
        "Multiple turns merged into a single message without proper spacing "
        "(for example, 'One moment please.Great, I found your account.')"
    ),
    "incomplete_message": "Messages cut off mid-sentence, fragments, or missing content.",
    "transcription_error": "Garbled text, speech-to-text errors, or voice transcription artifacts.",
    "repeated_message": "Duplicate or near-duplicate messages appearing within the conversation.",
    "incoherent_flow": (
        "Conversation flow breaks down: responses do not make sense in context, "
        "topic jumps are illogical, or questions are left unanswered."
    ),
    "poor_engagement": (
        "User provides only minimal responses throughout, with no substantive interaction."
    ),
    "authentication_loop": (
        "Conversation is stuck in identity verification without reaching actual problem-solving."
    ),
    "meaningless": "Conversation has no meaningful content or purpose for learning user behavior.",
}

CATEGORY_LIST = "\n".join(f"- **{key}**: {value}" for key, value in CATEGORIES.items())

SYSTEM_PROMPT = f"""You are evaluating customer-support conversation transcripts for training a user simulator.

Debug messages, DTMF markers, and SSML tags have already been stripped. Focus on remaining content quality.

Check this conversation against all categories below. A conversation is CLEAN only if it has zero issues.

## Issue Categories
{CATEGORY_LIST}

## Instructions
1. Read the entire conversation carefully.
2. Check every message for message-level issues.
3. Check the overall conversation for conversation-level issues.
4. Report all issues found. Be thorough but fair.
5. If no issues exist, mark the conversation as clean.

## Clarifications
- Company names and assistant names are not issues by themselves.
- Redaction markers like [PHONE], [NAME], [EMAIL], [ADDRESS], and [CREDIT-CARD] are expected.
- Short confirmations like "yes" and "okay" are normal individually.
- Flag poor_engagement only if the user provides virtually no substantive responses across the full conversation.
- Flag meaningless only for conversations that genuinely serve no training purpose."""


def load_conversations(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    conversations = list(data.values()) if isinstance(data, dict) else data
    normalized = []
    for i, conv in enumerate(conversations):
        messages = [
            {
                "speaker": str(msg.get("speaker", "")).strip() or "UNKNOWN",
                "text": str(msg.get("text", "")),
            }
            for msg in conv.get("messages", [])
            if str(msg.get("text", "")).strip()
        ]
        conv_id = str(conv.get("conversation_id") or conv.get("id") or f"row_{i:06d}")
        normalized.append(
            {
                **{k: v for k, v in conv.items() if k != "messages"},
                "conversation_id": conv_id,
                "messages": messages,
                "message_count": len(messages),
            }
        )
    return normalized


def preprocess_conversation(conv: dict[str, Any]) -> dict[str, Any]:
    cleaned = []
    for msg in conv["messages"]:
        text = msg["text"]
        if DEBUG_RE.match(text):
            continue
        text = SSML_RE.sub("", text)
        text = DTMF_RE.sub("", text)
        text = re.sub(r"  +", " ", text).strip()
        if not text:
            continue
        cleaned.append({"speaker": msg["speaker"], "text": text})
    return {
        **{k: v for k, v in conv.items() if k != "messages"},
        "conversation_id": conv["conversation_id"],
        "messages": cleaned,
        "message_count": len(cleaned),
        "original_message_count": len(conv["messages"]),
    }


def format_conversation(messages: list[dict[str, Any]]) -> str:
    return "\n".join(
        f"[{i}] {msg['speaker']}: {msg['text']}" for i, msg in enumerate(messages)
    )


def classify_one(text: str, model: str, client: Any) -> dict[str, Any]:
    from pydantic import BaseModel

    class QualityVerdict(BaseModel):
        is_clean: bool
        issues: list[str]
        reasoning: str

    response = client.responses.parse(
        model=model,
        instructions=SYSTEM_PROMPT,
        input=text,
        text_format=QualityVerdict,
    )
    verdict = response.output_parsed
    return {
        "model": model,
        "is_clean": verdict.is_clean,
        "issues": verdict.issues,
        "reasoning": verdict.reasoning,
    }


def process_batch(
    conversations: list[dict[str, Any]],
    client: Any,
    models: list[str],
    calls_per_model: int,
    majority_threshold: int,
    max_workers: int,
) -> list[dict[str, Any]]:
    prepped = {
        conv["conversation_id"]: preprocess_conversation(conv) for conv in conversations
    }
    texts = {cid: format_conversation(prepped[cid]["messages"]) for cid in prepped}
    tasks = [
        (cid, model)
        for cid in prepped
        for model in models
        for _ in range(calls_per_model)
    ]
    votes_by_conv: dict[str, list[dict[str, Any]]] = {cid: [] for cid in prepped}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(classify_one, texts[cid], model, client): (cid, model)
            for cid, model in tasks
        }
        for future in as_completed(futures):
            cid, model = futures[future]
            try:
                votes_by_conv[cid].append(future.result())
            except Exception as exc:
                votes_by_conv[cid].append(
                    {
                        "model": model,
                        "is_clean": False,
                        "issues": ["api_error"],
                        "reasoning": str(exc),
                    }
                )

    results = []
    for cid, prepped_conv in prepped.items():
        votes = votes_by_conv[cid]
        clean_count = sum(1 for vote in votes if vote["is_clean"])
        issue_counts: Counter[str] = Counter()
        for vote in votes:
            issue_counts.update(vote.get("issues", []))
        is_clean = clean_count >= majority_threshold
        results.append(
            {
                "conversation_id": cid,
                "message_count": prepped_conv["message_count"],
                "original_message_count": prepped_conv["original_message_count"],
                "is_clean": is_clean,
                "clean_votes": clean_count,
                "total_votes": len(votes),
                "issue_counts": dict(issue_counts.most_common()),
                "votes": votes,
                "messages": prepped_conv["messages"] if is_clean else None,
            }
        )
    return results


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def load_progress(path: Path | None) -> list[dict[str, Any]]:
    if path and path.exists():
        return json.loads(path.read_text())
    return []


def save_progress(path: Path | None, results: list[dict[str, Any]]) -> None:
    if not path:
        return
    progress = [
        {key: value for key, value in result.items() if key != "messages"}
        for result in results
    ]
    write_json(path, progress)


def run_filter(args: argparse.Namespace) -> dict[str, Any]:
    import openai

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    client = openai.OpenAI(api_key=api_key)

    conversations = load_conversations(args.input)
    if args.limit:
        conversations = conversations[: args.limit]
    random.seed(args.seed)
    random.shuffle(conversations)

    all_results = load_progress(args.progress_file)
    processed_ids = {row["conversation_id"] for row in all_results}
    clean_count = sum(1 for row in all_results if row["is_clean"])
    remaining = [
        conv for conv in conversations if conv["conversation_id"] not in processed_ids
    ]

    models = [model.strip() for model in args.models.split(",") if model.strip()]
    total_votes = len(models) * args.calls_per_model
    majority_threshold = args.threshold or (total_votes // 2 + 1)

    for i in range(0, len(remaining), args.batch_size):
        if clean_count >= args.target_clean:
            break
        batch = remaining[i : i + args.batch_size]
        print(
            f"batch {i // args.batch_size + 1}: {len(batch)} conversations | clean={clean_count}/{args.target_clean}"
        )
        batch_results = process_batch(
            batch,
            client,
            models,
            args.calls_per_model,
            majority_threshold,
            args.max_workers,
        )
        all_results.extend(batch_results)
        clean_count += sum(1 for row in batch_results if row["is_clean"])
        save_progress(args.progress_file, all_results)

    clean_ids = {row["conversation_id"] for row in all_results if row["is_clean"]}
    clean_output = []
    for conv in conversations:
        if (
            conv["conversation_id"] in clean_ids
            and len(clean_output) < args.target_clean
        ):
            prepped = preprocess_conversation(conv)
            clean_output.append(
                {
                    "conversation_id": prepped["conversation_id"],
                    "message_count": prepped["message_count"],
                    "messages": prepped["messages"],
                }
            )

    full_output = [
        {key: value for key, value in row.items() if key != "messages"}
        for row in all_results
    ]
    write_json(args.output_clean, clean_output)
    write_json(args.output_full, full_output)

    issue_counts: Counter[str] = Counter()
    dirty = [row for row in full_output if not row["is_clean"]]
    for row in dirty:
        issue_counts.update(row.get("issue_counts", {}))
    return {
        "processed": len(all_results),
        "clean": len(clean_output),
        "target_clean": args.target_clean,
        "models": models,
        "calls_per_model": args.calls_per_model,
        "threshold": majority_threshold,
        "dirty_issue_counts": dict(issue_counts.most_common()),
    }


def dry_run(args: argparse.Namespace) -> None:
    conversations = load_conversations(args.input)
    if args.limit:
        conversations = conversations[: args.limit]
    print(f"loaded={len(conversations)}")
    if conversations:
        prepped = preprocess_conversation(conversations[0])
        print(f"first_conversation_id={prepped['conversation_id']}")
        print(
            f"messages={prepped['message_count']} original_messages={prepped['original_message_count']}"
        )
        print(format_conversation(prepped["messages"])[:1500])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="raw or rule-cleaned conversation JSON",
    )
    parser.add_argument("--output-clean", required=True, type=Path)
    parser.add_argument("--output-full", required=True, type=Path)
    parser.add_argument("--progress-file", type=Path)
    parser.add_argument("--models", default="gpt-5-mini,gpt-4.1-mini,gpt-5.4-mini")
    parser.add_argument("--calls-per-model", type=int, default=2)
    parser.add_argument(
        "--threshold",
        type=int,
        default=None,
        help="clean votes needed; default is strict majority",
    )
    parser.add_argument("--target-clean", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--max-workers", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--limit", type=int, default=None, help="debug limit on input conversations"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="preprocess and print one example without calling the API",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dry_run:
        dry_run(args)
        return
    summary = run_filter(args)
    print("LLM quality filtering complete")
    print(
        f"processed={summary['processed']} clean={summary['clean']}/{summary['target_clean']} "
        f"threshold={summary['threshold']}"
    )


if __name__ == "__main__":
    main()
