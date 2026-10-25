#!/usr/bin/env python3
"""Deterministic conversation cleaning for persona induction.

This file-based pipeline adapts the Brinks data-cleaning stages from the
`virtual_agent/notebooks/retrieval` workflow without depending on Cresta fetch
SDKs. It starts from an exported raw conversation JSON and writes cleaned
conversation files that can be passed to
`cresta.persona.induction.extract_seed_personas`.

Expected input:
  JSON list, or JSON object whose values are conversations, with each record:
    {
      "conversation_id": "...",
      "messages": [{"speaker": "VISITOR"|"AGENT"|"BOT", "text": "..."}, ...]
    }

Example:
  .venv/bin/python -m cresta.persona.data_cleaning.rule_cleaning \
    --input /path/to/raw_10k_conversations.json \
    --out-dir /path/to/persona-state/brinks_rule
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

TIER12_CATEGORIES = {
    "debug_message",
    "dtmf_artifact",
    "ssml_markup",
    "markup_artifact",
    "unredacted_pii",
    "incomplete_message",
    "repeated_message",
}
STRIPPABLE_CATEGORIES = {
    "debug_message",
    "dtmf_artifact",
    "ssml_markup",
    "markup_artifact",
}

FILLER_WORDS = {
    "yes",
    "yes.",
    "no",
    "no.",
    "yeah",
    "yeah.",
    "yep",
    "nope",
    "okay",
    "okay.",
    "ok",
    "ok.",
    "correct",
    "correct.",
    "that's correct",
    "that's correct.",
    "that's it",
    "that's it.",
    "that is correct",
    "that is correct.",
    "thank you",
    "thank you.",
    "thanks",
    "thanks.",
    "goodbye",
    "goodbye.",
    "bye",
    "bye.",
}

TOPIC_KEYWORDS = {
    "camera/video": ["camera", "video", "recording", "clip", "footage"],
    "billing/payment": ["payment", "bill", "balance", "charge", "autopay", "pay"],
    "alarm/sensor": ["alarm", "sensor", "dispatch", "alert", "motion"],
    "cancellation": ["cancel"],
    "scheduling/tech": ["schedule", "appointment", "technician", "tech visit"],
    "wifi/network": ["wi-fi", "wifi", "internet", "router", "network"],
    "installation": ["installation", "install", "agreement", "monthly monitoring"],
    "account/email": ["paperless", "email address", "account information", "password"],
}

SSML_RE = re.compile(
    r"</?(?:break|speak|prosody|say-as|phoneme|sub|voice|audio)[^>]*/?>", re.I
)
DTMF_RE = re.compile(r"\[/?DTMF[^\]]*\]", re.I)


def load_conversations(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    conversations = list(data.values()) if isinstance(data, dict) else data
    normalized = []
    seen_ids: set[str] = set()
    for i, conv in enumerate(conversations):
        messages = [
            {
                "speaker": str(msg.get("speaker", "")).strip() or "UNKNOWN",
                "text": str(msg.get("text", "")),
                **(
                    {"redaction_state": msg["redaction_state"]}
                    if "redaction_state" in msg
                    else {}
                ),
            }
            for msg in conv.get("messages", [])
            if str(msg.get("text", "")).strip()
        ]
        conv_id = str(conv.get("conversation_id") or conv.get("id") or f"row_{i:06d}")
        if conv_id in seen_ids:
            continue
        seen_ids.add(conv_id)
        normalized.append(
            {
                **{k: v for k, v in conv.items() if k != "messages"},
                "conversation_id": conv_id,
                "messages": messages,
                "message_count": len(messages),
            }
        )
    return normalized


def detect_debug_message(text: str, msg_index: int, speaker: str) -> bool:
    lower = text.lower()
    if "for debugging" in lower:
        return True
    if msg_index <= 2 and speaker == "AGENT" and "ani number" in lower:
        return True
    return False


def detect_markup_artifact(text: str) -> bool:
    if DTMF_RE.search(text):
        return False
    if "[Custom AI Agent Bank Accounts]" in text:
        return True
    if re.search(r"\[\[.+?\]\]", text):
        return True
    if re.search(
        r"\[(?!CREDIT-CARD\]|PHONE\]|NAME\]|EMAIL\]|ADDRESS\])[A-Z][A-Z-]+\]", text
    ):
        if re.search(
            r"\[CREDIT-CARD-EXPIRATION\]|\[CVV\]|\[ROUTING-NUMBER\]|\[ACCOUNT-NUMBER\]",
            text,
        ):
            return True
    return False


def detect_placeholder_artifact(text: str) -> bool:
    lower = text.lower()
    stripped_lower = text.strip().lower()
    patterns = [
        r"(?:code\s*word|codeword)\s+is\s+(?:code\s*word|codeword)",
        r"^(?:my\s+)?(?:code\s*word|codeword)s?[\s.!?]*$",
        r"[CB]0DEW0RD",
        r"test.*(?:at\s+cresta|cresta\s*\.)",
        r"\[\[(?:CREDIT-CARD|CVV|SSN)",
        r"verbal password listed as .{0,5}tester",
        r"(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\w+,?\s*nineteen hundred\b",
    ]
    return any(
        re.search(pattern, lower if pattern != r"[CB]0DEW0RD" else text)
        for pattern in patterns
    ) or bool(re.match(patterns[1], stripped_lower))


def detect_unredacted_pii(text: str, speaker: str) -> bool:
    lower = text.lower()
    if speaker == "AGENT":
        agent_patterns = [
            r"(?:your\s+)?address\s+as\b",
            r"(?:phone|primary)\s+number\s+as\b",
            r"first number contacted",
            r"(?:primary\s+)?email\s+as\b",
            r"at\s+(?:gmail|yahoo|hotmail|outlook|aol|icloud|brinkshome)\s+dot\s+com",
            r"verbal password listed as",
            r"(?:date of birth|birthday)\s+is\s+\w",
        ]
        if any(re.search(pattern, lower) for pattern in agent_patterns):
            return True
        if re.search(r"emergency contact", lower) and re.search(r"at\s+\w", lower):
            return True
        if re.search(
            r"(?:speaking with|confirm.*am I speaking)\s+[A-Z][a-z]{2,}", text
        ):
            return True
    if speaker == "VISITOR":
        visitor_patterns = [
            r"\b\d{10,}\b",
            r"\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}",
            r"\b\d(?:\s+\d){6,}\b",
            r"\b\d{2}/\d{2}/\d{4}\b",
        ]
        return any(re.search(pattern, text) for pattern in visitor_patterns)
    return False


def detect_incomplete_message(text: str, speaker: str) -> bool:
    stripped = text.strip()
    if len(stripped) < 3:
        return True
    if text.startswith(" ") and len(stripped) < 60:
        return True
    if len(stripped) > 10 and re.search(
        r"\b(?:the|a|an|to|of|for|with|on|or|and|in|your|my|by|you|would|Security|that)\s*$",
        stripped,
    ):
        return True
    if speaker == "AGENT" and len(stripped) > 20 and stripped.endswith(","):
        return True
    return False


def detect_repeated_message(
    text: str, all_messages: list[dict[str, Any]], msg_index: int, speaker: str
) -> bool:
    if len(text.strip()) < 20:
        return False
    if len(text) > 80:
        half = len(text) // 2
        if SequenceMatcher(None, text[:half], text[half:]).ratio() > 0.8:
            return True
    for j in range(max(0, msg_index - 3), min(len(all_messages), msg_index + 4)):
        if j == msg_index:
            continue
        other = all_messages[j]
        if other["speaker"] != speaker:
            continue
        other_text = other["text"]
        if text == other_text and len(text) > 20:
            return True
        if len(text) > 40 and len(other_text) > 40:
            if SequenceMatcher(None, text, other_text).ratio() > 0.90:
                return True
    return False


def detect_concatenated_turns(text: str) -> bool:
    return len(text) >= 20 and bool(re.search(r"[.!?][A-Z]", text))


def detect_conversation(conv: dict[str, Any]) -> dict[str, Any]:
    annotations = []
    messages = conv["messages"]
    for i, msg in enumerate(messages):
        text = msg["text"]
        speaker = msg["speaker"]
        categories = []
        if detect_debug_message(text, i, speaker):
            categories.append("debug_message")
        if DTMF_RE.search(text):
            categories.append("dtmf_artifact")
        if SSML_RE.search(text):
            categories.append("ssml_markup")
        if detect_markup_artifact(text) or detect_placeholder_artifact(text):
            categories.append("markup_artifact")
        if detect_unredacted_pii(text, speaker):
            categories.append("unredacted_pii")
        if detect_concatenated_turns(text):
            categories.append("concatenated_turns")
        if detect_incomplete_message(text, speaker):
            categories.append("incomplete_message")
        if detect_repeated_message(text, messages, i, speaker):
            categories.append("repeated_message")

        filtered = [cat for cat in categories if cat in TIER12_CATEGORIES]
        if filtered:
            annotations.append({"message_index": i, "categories": filtered})
    return {
        "conversation_id": conv["conversation_id"],
        "message_count": conv["message_count"],
        "annotations": annotations,
    }


def strip_flagged_messages(
    conv: dict[str, Any], annotations: list[dict[str, Any]]
) -> dict[str, Any]:
    strip_indices = {
        ann["message_index"]
        for ann in annotations
        if any(category in STRIPPABLE_CATEGORIES for category in ann["categories"])
    }
    if not strip_indices:
        return conv
    new_messages = [
        msg for i, msg in enumerate(conv["messages"]) if i not in strip_indices
    ]
    return {**conv, "messages": new_messages, "message_count": len(new_messages)}


def fix_concatenated_turns(conv: dict[str, Any]) -> dict[str, Any]:
    new_messages = []
    for msg in conv["messages"]:
        if msg["speaker"] == "AGENT" and detect_concatenated_turns(msg["text"]):
            parts = [
                part.strip()
                for part in re.split(r"(?<=[.!?])(?=[A-Z])", msg["text"])
                if part.strip()
            ]
            if len(parts) > 1:
                new_messages.extend({**msg, "text": part} for part in parts)
                continue
        new_messages.append(msg)
    return {**conv, "messages": new_messages, "message_count": len(new_messages)}


def fix_ssml_tags(conv: dict[str, Any]) -> dict[str, Any]:
    new_messages = []
    for msg in conv["messages"]:
        text = SSML_RE.sub("", msg["text"]).strip()
        if text:
            new_messages.append({**msg, "text": text})
    return {**conv, "messages": new_messages, "message_count": len(new_messages)}


def merge_adjacent_fragments(conv: dict[str, Any]) -> dict[str, Any]:
    messages = conv["messages"]
    if len(messages) <= 1:
        return conv
    merged = [messages[0].copy()]
    for curr in messages[1:]:
        prev = merged[-1]
        if (
            curr["text"].startswith(" ")
            and len(curr["text"].strip()) < 60
            and curr["speaker"] == prev["speaker"]
        ):
            prev["text"] = prev["text"].rstrip() + " " + curr["text"].strip()
        else:
            merged.append(curr.copy())
    return {**conv, "messages": merged, "message_count": len(merged)}


def drop_residual_messages(conv: dict[str, Any]) -> dict[str, Any]:
    keep = []
    for msg in conv["messages"]:
        text = msg["text"].strip()
        if len(text) < 3:
            continue
        if text == ",":
            continue
        if msg["text"].startswith(" ") and len(text) < 10:
            continue
        keep.append({**msg, "text": text})
    return {**conv, "messages": keep, "message_count": len(keep)}


def iterative_debug_cleanup(conv: dict[str, Any]) -> dict[str, Any]:
    for _ in range(20):
        result = detect_conversation(conv)
        debug_indices = {
            ann["message_index"]
            for ann in result["annotations"]
            if "debug_message" in ann["categories"]
        }
        if not debug_indices:
            break
        new_messages = [
            msg for i, msg in enumerate(conv["messages"]) if i not in debug_indices
        ]
        if len(new_messages) == len(conv["messages"]):
            break
        conv = {**conv, "messages": new_messages, "message_count": len(new_messages)}
    return conv


def clean_one(
    conv: dict[str, Any], annotations: list[dict[str, Any]]
) -> dict[str, Any]:
    cleaned = strip_flagged_messages(conv, annotations)
    cleaned = fix_concatenated_turns(cleaned)
    cleaned = fix_ssml_tags(cleaned)
    cleaned = merge_adjacent_fragments(cleaned)
    cleaned = drop_residual_messages(cleaned)
    return iterative_debug_cleanup(cleaned)


def clean_conversations(
    conversations: list[dict[str, Any]],
    detection_results: list[dict[str, Any]],
    min_messages: int,
) -> list[dict[str, Any]]:
    detection_map = {
        row["conversation_id"]: row["annotations"] for row in detection_results
    }
    cleaned = []
    for conv in conversations:
        row = clean_one(conv, detection_map.get(conv["conversation_id"], []))
        if row["message_count"] >= min_messages:
            cleaned.append(row)
    return cleaned


def length_filter(
    conversations: list[dict[str, Any]], lower_pct: int, upper_pct: int
) -> list[dict[str, Any]]:
    if not conversations:
        return []
    counts = sorted(conv["message_count"] for conv in conversations)
    n = len(counts)
    low = counts[min(n - 1, max(0, n * lower_pct // 100))]
    high = counts[min(n - 1, max(0, n * upper_pct // 100))]
    return [conv for conv in conversations if low <= conv["message_count"] <= high]


def apply_hard_filters(conv: dict[str, Any]) -> str | None:
    messages = conv["messages"]
    speakers = {msg["speaker"] for msg in messages}
    if speakers == {"AGENT"}:
        return "agent_only"
    if speakers == {"VISITOR"}:
        return "visitor_only"
    if len(messages) >= 4 and all(msg["speaker"] == "AGENT" for msg in messages[-4:]):
        return "dead_ending"

    visitor_count = sum(1 for msg in messages if msg["speaker"] == "VISITOR")
    if messages and visitor_count / len(messages) < 0.2:
        return "speaker_imbalance"

    auth_done = False
    post_auth_visitor = 0
    for msg in messages:
        if "authenticated" in msg["text"].lower():
            auth_done = True
        elif auth_done and msg["speaker"] == "VISITOR":
            post_auth_visitor += 1
    if auth_done and post_auth_visitor == 0:
        return "auth_only"

    text_counts = Counter(msg["text"] for msg in messages if len(msg["text"]) > 20)
    if text_counts and max(text_counts.values()) >= 3:
        return "repeated_loop"

    post_greeting_turns = 0
    past_greeting = False
    for msg in messages:
        lower = msg["text"].lower()
        if "how can we help" in lower or "authenticated" in lower:
            past_greeting = True
            continue
        if past_greeting:
            post_greeting_turns += 1
    if past_greeting and post_greeting_turns < 4:
        return "too_few_substantive_turns"
    return None


def classify_topics(conv: dict[str, Any]) -> list[str]:
    all_text = " ".join(msg["text"].lower() for msg in conv["messages"])
    return [
        topic
        for topic, keywords in TOPIC_KEYWORDS.items()
        if any(kw in all_text for kw in keywords)
    ]


def classify_conv_type(conv: dict[str, Any]) -> str:
    all_text = " ".join(msg["text"].lower() for msg in conv["messages"])
    if any(
        keyword in all_text
        for keyword in [
            "chose brinks home",
            "sixty month agreement",
            "installation is complete",
            "monthly monitoring rate",
        ]
    ):
        return "welcome"
    if any(
        keyword in all_text
        for keyword in ["authenticated", "how can we help", "codeword"]
    ):
        return "care"
    return "other"


def classify_resolution(conv: dict[str, Any]) -> str:
    all_text = " ".join(msg["text"].lower() for msg in conv["messages"])
    has_transfer = any(
        keyword in all_text
        for keyword in ["connect you with", "hold while i connect", "transfer"]
    )
    has_farewell = any(
        keyword in all_text
        for keyword in ["pleasure speaking", "thank you for calling", "have a great"]
    )
    has_disconnect = any(
        keyword in all_text for keyword in ["call me back", "disconnect", "still there"]
    )
    if has_farewell and has_transfer:
        return "resolved_then_transfer"
    if has_farewell:
        return "resolved_by_bot"
    if has_transfer:
        return "transfer_to_human"
    if has_disconnect:
        return "timeout"
    return "unclear"


def score_conversation(conv: dict[str, Any]) -> dict[str, float]:
    messages = conv["messages"]
    all_text = " ".join(msg["text"].lower() for msg in messages)
    has_greeting = any(
        "veronica" in msg["text"].lower() or "brinks home" in msg["text"].lower()
        for msg in messages[:3]
    )
    has_auth = "authenticated" in all_text or "codeword" in all_text
    has_farewell = any(
        keyword in all_text
        for keyword in [
            "pleasure speaking",
            "thank you for calling",
            "hold while i connect",
        ]
    )
    completeness = sum([has_greeting, has_auth, has_farewell]) / 3
    alternations = sum(
        1
        for i in range(1, len(messages))
        if messages[i]["speaker"] != messages[i - 1]["speaker"]
    )
    turn_taking = alternations / max(len(messages) - 1, 1)
    visitor_msgs = [
        msg["text"].strip().lower().rstrip(".!?")
        for msg in messages
        if msg["speaker"] == "VISITOR"
    ]
    info_density = (
        sum(1 for text in visitor_msgs if text not in FILLER_WORDS and len(text) > 5)
        / len(visitor_msgs)
        if visitor_msgs
        else 0
    )
    overall = completeness * 0.4 + turn_taking * 0.3 + info_density * 0.3
    return {
        "completeness": completeness,
        "turn_taking": turn_taking,
        "info_density": info_density,
        "overall": overall,
    }


def benchmark_selection(
    conversations: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], Counter[str]]:
    candidates = []
    rejected: Counter[str] = Counter()
    for conv in conversations:
        reason = apply_hard_filters(conv)
        if reason:
            rejected[reason] += 1
            continue
        message_count = conv["message_count"]
        candidates.append(
            {
                "conversation_id": conv["conversation_id"],
                "message_count": message_count,
                "messages": conv["messages"],
                "conv_type": classify_conv_type(conv),
                "resolution": classify_resolution(conv),
                "topics": classify_topics(conv),
                "length_bucket": "short"
                if message_count <= 10
                else "medium"
                if message_count <= 20
                else "long",
                "scores": score_conversation(conv),
            }
        )
    return candidates, rejected


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def build_report(
    raw: list[dict[str, Any]],
    detection: list[dict[str, Any]],
    cleaned: list[dict[str, Any]],
    filtered: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    rejected: Counter[str],
) -> dict[str, Any]:
    category_counts: Counter[str] = Counter()
    for result in detection:
        for ann in result["annotations"]:
            category_counts.update(ann["categories"])
    return {
        "summary": {
            "raw": len(raw),
            "cleaned": len(cleaned),
            "length_filtered": len(filtered),
            "benchmark_candidates": len(candidates),
        },
        "rule_detection_categories": dict(category_counts.most_common()),
        "benchmark_rejections": dict(rejected.most_common()),
    }


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = args.out_dir
    raw = load_conversations(args.input)
    detection = [detect_conversation(conv) for conv in raw]
    cleaned = clean_conversations(raw, detection, args.min_messages)
    filtered = (
        cleaned
        if args.no_length_filter
        else length_filter(cleaned, args.lower_pct, args.upper_pct)
    )
    candidates, rejected = benchmark_selection(filtered)
    report = build_report(raw, detection, cleaned, filtered, candidates, rejected)

    write_json(out_dir / "rule_detection.json", detection)
    write_json(out_dir / "cleaned_conversations.json", cleaned)
    write_json(out_dir / "cleaned_conversations_filtered.json", filtered)
    write_json(out_dir / "benchmark_candidates.json", candidates)
    write_json(out_dir / "report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input", required=True, type=Path, help="raw exported conversation JSON"
    )
    parser.add_argument(
        "--out-dir", required=True, type=Path, help="directory for pipeline outputs"
    )
    parser.add_argument("--min-messages", type=int, default=2)
    parser.add_argument(
        "--lower-pct",
        type=int,
        default=10,
        help="lower message-count percentile for length filter",
    )
    parser.add_argument(
        "--upper-pct",
        type=int,
        default=90,
        help="upper message-count percentile for length filter",
    )
    parser.add_argument("--no-length-filter", action="store_true")
    return parser.parse_args()


def main() -> None:
    report = run_pipeline(parse_args())
    summary = report["summary"]
    print("Data cleaning complete")
    print(
        f"raw={summary['raw']} cleaned={summary['cleaned']} "
        f"length_filtered={summary['length_filtered']} candidates={summary['benchmark_candidates']}"
    )


if __name__ == "__main__":
    main()
