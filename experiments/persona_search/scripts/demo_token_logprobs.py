#!/usr/bin/env python3
"""Demo: inspect output-token logprobs from OpenAI or Vertex Gemini.

This demonstrates what hosted APIs expose:
  - the logprob of each token the model actually emitted
  - up to top-k alternative tokens at each output position

They do NOT expose full-vocabulary logits/probabilities.

Examples:
  # OpenAI Responses API
  OPENAI_API_KEY=... OPENAI_BASE_URL=... python experiments/persona_search/scripts/demo_token_logprobs.py \
      --provider openai --model gpt-4.1-mini --top-k 5

  # Vertex / Gemini via google-genai
  CRESTA_VERTEX_TOKEN=... VERTEXAI_PROJECT=cresta-ml VERTEXAI_LOCATION=us-central1 \
  python experiments/persona_search/scripts/demo_token_logprobs.py \
      --provider vertex --model gemini-1.5-pro --top-k 5
  # Note: the current Cresta /proxy/vertex deployment may only allow Anthropic
  # Vertex models; if Gemini is not allowlisted, use --provider openai or gemini.

  # Gemini Developer API via API key
  GEMINI_API_KEY=... python experiments/persona_search/scripts/demo_token_logprobs.py \
      --provider gemini --model gemini-2.5-flash --top-k 5

  # Raw Vertex REST through Cresta dev proxy (best diagnostic for proxy support)
  CRESTA_VERTEX_TOKEN=... python experiments/persona_search/scripts/demo_token_logprobs.py \
      --provider vertex-proxy-raw --model gemini-2.0-flash-001 --top-k 5
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))


PROMPT = 'Repeat exactly this sentence and nothing else: "The cat sat."'


@dataclass
class TokenAlt:
    token: str
    logprob: float

    @property
    def prob(self) -> float:
        return math.exp(self.logprob)


def _norm_label(token: str) -> str:
    return token.strip().upper()


def _print_position(
    chosen: TokenAlt, alternatives: list[TokenAlt], labels: tuple[str, ...] = ()
) -> None:
    print(
        f"chosen token: {chosen.token!r}  logprob={chosen.logprob:.4f}  p={chosen.prob:.4f}"
    )
    print("\ntop alternatives:")
    for alt in alternatives:
        print(f"  {alt.token!r:<12} logprob={alt.logprob:>8.4f}  p={alt.prob:.4f}")

    label_scores = {}
    for alt in alternatives:
        lab = _norm_label(alt.token)
        if lab in labels:
            label_scores[lab] = alt.prob
    if labels and label_scores:
        z = sum(label_scores.values())
        print("\nlabel probs normalized over returned label tokens only:")
        for lab in labels:
            if lab in label_scores:
                print(f"  {lab}: {label_scores[lab] / z:.4f}")
            else:
                print(f"  {lab}: not returned in top-k")


def _print_all_positions(logprobs: list[dict], labels: tuple[str, ...] = ()) -> None:
    for i, lp in enumerate(logprobs):
        chosen = TokenAlt(lp["token"], lp["logprob"])
        alts = [TokenAlt(x["token"], x["logprob"]) for x in lp.get("top_logprobs", [])]
        print(f"\n=== output token position {i} ===")
        _print_position(chosen, alts, labels)


def run_openai(
    model: str,
    top_k: int,
    prompt: str,
    max_output_tokens: int,
    base_url: str | None,
) -> None:
    try:
        from openai import OpenAI
    except ImportError as e:
        raise SystemExit("Missing dependency: pip install openai") from e

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY first.")

    client = OpenAI(base_url=base_url) if base_url else OpenAI()
    # Responses API currently enforces a minimum max_output_tokens of 16.
    resp = client.responses.create(
        model=model,
        input=prompt,
        temperature=0,
        max_output_tokens=max(max_output_tokens, 16),
        top_logprobs=top_k,
        include=["message.output_text.logprobs"],
    )

    # Responses API returns output items; find the first output_text content block
    # with logprobs. Object shapes can vary slightly across SDK versions, so use
    # model_dump() for a plain dict.
    d = resp.model_dump()
    for item in d.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text" and content.get("logprobs"):
                print("provider=openai")
                print(f"model={model}")
                if base_url:
                    print(f"base_url={base_url}")
                print(f"prompt={prompt!r}")
                print(f"output_text={content.get('text')!r}")
                _print_all_positions(content["logprobs"])
                return
    raise SystemExit("No output_text.logprobs found in the response.")


def run_vertex(
    model: str,
    top_k: int,
    prompt: str,
    max_output_tokens: int,
    vertex_auth: str,
) -> None:
    try:
        from google import genai
        from google.genai import types
    except ImportError as e:
        raise SystemExit("Missing dependency: pip install google-genai") from e

    # In this repo, Vertex is often reached through the Cresta dev proxy, not
    # Google ADC. The patch below:
    #   1. makes google.auth.default return static token credentials, and
    #   2. rewrites *-aiplatform.googleapis.com URLs to CRESTA_VERTEX_PROXY_BASE.
    #
    # Existing scripts set CRESTA_VERTEX_TOKEN from ANTHROPIC_API_KEY, so mirror
    # that convenience here.
    if (
        vertex_auth == "cresta"
        and not os.environ.get("CRESTA_VERTEX_TOKEN")
        and os.environ.get("ANTHROPIC_API_KEY")
    ):
        os.environ["CRESTA_VERTEX_TOKEN"] = os.environ["ANTHROPIC_API_KEY"]
    if vertex_auth == "cresta" and os.environ.get("CRESTA_VERTEX_TOKEN"):
        import tau2.utils.cresta_vertex  # noqa: F401, PLC0415

    project = (
        os.environ.get("GOOGLE_CLOUD_PROJECT")
        or os.environ.get("VERTEXAI_PROJECT")
        or (
            "cresta-ml"
            if vertex_auth == "cresta" and os.environ.get("CRESTA_VERTEX_TOKEN")
            else None
        )
    )
    location = (
        os.environ.get("GOOGLE_CLOUD_LOCATION")
        or os.environ.get("VERTEXAI_LOCATION")
        or "us-central1"
    )
    if not project:
        raise SystemExit(
            "Set GOOGLE_CLOUD_PROJECT for --vertex-auth adc, or CRESTA_VERTEX_TOKEN "
            "for --vertex-auth cresta."
        )

    client = genai.Client(vertexai=True, project=project, location=location)
    resp = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0,
            max_output_tokens=max_output_tokens,
            response_logprobs=True,
            logprobs=top_k,
        ),
    )

    cand = resp.candidates[0]
    lpr = cand.logprobs_result
    print("provider=vertex")
    print(f"vertex_auth={vertex_auth}")
    print(f"model={model}")
    if os.environ.get("CRESTA_VERTEX_TOKEN"):
        print(
            f"proxy_base={os.environ.get('CRESTA_VERTEX_PROXY_BASE', 'https://dev-proxy.ops2.internal.cresta.ai/proxy/vertex')}"
        )
    print(f"project={project} location={location}")
    print(f"prompt={prompt!r}")
    print(f"output_text={resp.text!r}")
    for i, chosen0 in enumerate(lpr.chosen_candidates):
        top = lpr.top_candidates[i].candidates if i < len(lpr.top_candidates) else []
        chosen = TokenAlt(chosen0.token, chosen0.log_probability)
        alts = [TokenAlt(x.token, x.log_probability) for x in top]
        print(f"\n=== output token position {i} ===")
        _print_position(chosen, alts)


def run_gemini_api_key(
    model: str, top_k: int, prompt: str, max_output_tokens: int
) -> None:
    try:
        from google import genai
        from google.genai import types
    except ImportError as e:
        raise SystemExit("Missing dependency: pip install google-genai") from e

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("Set GEMINI_API_KEY or GOOGLE_API_KEY first.")

    client = genai.Client(api_key=api_key)
    resp = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0,
            max_output_tokens=max_output_tokens,
            response_logprobs=True,
            logprobs=top_k,
        ),
    )

    cand = resp.candidates[0]
    lpr = cand.logprobs_result
    print("provider=gemini")
    print(f"model={model}")
    print(f"prompt={prompt!r}")
    print(f"output_text={resp.text!r}")
    for i, chosen0 in enumerate(lpr.chosen_candidates):
        top = lpr.top_candidates[i].candidates if i < len(lpr.top_candidates) else []
        chosen = TokenAlt(chosen0.token, chosen0.log_probability)
        alts = [TokenAlt(x.token, x.log_probability) for x in top]
        print(f"\n=== output token position {i} ===")
        _print_position(chosen, alts)


def run_vertex_proxy_raw(
    model: str, top_k: int, prompt: str, max_output_tokens: int
) -> None:
    """Call Vertex REST through the Cresta dev proxy directly.

    This bypasses google-genai and the httpx rewrite patch so we can see whether
    the proxy itself allows Google/Gemini publisher models and logprobs.
    """
    import urllib.error
    import urllib.request

    token = os.environ.get("CRESTA_VERTEX_TOKEN") or os.environ.get("ANTHROPIC_API_KEY")
    if not token:
        raise SystemExit("Set CRESTA_VERTEX_TOKEN (or ANTHROPIC_API_KEY) first.")

    base = os.environ.get(
        "CRESTA_VERTEX_PROXY_BASE",
        "https://dev-proxy.ops2.internal.cresta.ai/proxy/vertex",
    ).rstrip("/")
    project = os.environ.get("VERTEXAI_PROJECT", "cresta-ml")
    location = os.environ.get("VERTEXAI_LOCATION", "us-central1")
    url = (
        f"{base}/v1/projects/{project}/locations/{location}"
        f"/publishers/google/models/{model}:generateContent"
    )
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": max_output_tokens,
            "responseLogprobs": True,
            "logprobs": top_k,
        },
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        print("provider=vertex-proxy-raw")
        print(f"model={model}")
        print(f"url={url}")
        print(f"HTTP {e.code}")
        print(body)
        raise SystemExit(e.code) from e

    cand = body["candidates"][0]
    lpr = cand.get("logprobsResult")
    print("provider=vertex-proxy-raw")
    print(f"model={model}")
    print(f"url={url}")
    text = "".join(p.get("text", "") for p in cand.get("content", {}).get("parts", []))
    print(f"prompt={prompt!r}")
    print(f"output_text={text!r}")
    if not lpr:
        print("No logprobsResult returned. Response keys:", sorted(cand))
        return
    chosen = lpr.get("chosenCandidates", [])
    top = lpr.get("topCandidates", [])
    for i, chosen0 in enumerate(chosen):
        chosen_alt = TokenAlt(chosen0["token"], chosen0["logProbability"])
        top_alts = [
            TokenAlt(x["token"], x["logProbability"])
            for x in (top[i].get("candidates", []) if i < len(top) else [])
        ]
        print(f"\n=== output token position {i} ===")
        _print_position(chosen_alt, top_alts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--provider",
        choices=["openai", "vertex", "gemini", "vertex-proxy-raw"],
        required=True,
    )
    ap.add_argument(
        "--model",
        default=None,
        help=(
            "Model name. Defaults: openai=gpt-4.1-mini, "
            "vertex=gemini-1.5-pro (only if the dev proxy allowlists Gemini), "
            "gemini=gemini-2.5-flash."
        ),
    )
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--max-output-tokens", type=int, default=16)
    ap.add_argument(
        "--vertex-auth",
        choices=["adc", "cresta"],
        default="adc",
        help=(
            "Vertex auth route. Use adc for real Vertex Gemini logprobs; use cresta "
            "for the Cresta dev proxy (usually Anthropic-on-Vertex, often not useful "
            "for Gemini logprobs)."
        ),
    )
    ap.add_argument(
        "--openai-base-url",
        default=os.environ.get("OPENAI_BASE_URL"),
        help="OpenAI-compatible proxy base URL. Defaults to OPENAI_BASE_URL.",
    )
    args = ap.parse_args()

    if not 1 <= args.top_k <= 20:
        raise SystemExit("--top-k must be between 1 and 20 for hosted APIs.")

    if args.provider == "openai":
        run_openai(
            args.model or "gpt-4.1-mini",
            args.top_k,
            args.prompt,
            args.max_output_tokens,
            args.openai_base_url,
        )
    elif args.provider == "vertex":
        run_vertex(
            args.model or "gemini-2.5-flash",
            args.top_k,
            args.prompt,
            args.max_output_tokens,
            args.vertex_auth,
        )
    elif args.provider == "gemini":
        run_gemini_api_key(
            args.model or "gemini-2.5-flash",
            args.top_k,
            args.prompt,
            args.max_output_tokens,
        )
    else:
        run_vertex_proxy_raw(
            args.model or "gemini-2.0-flash-001",
            args.top_k,
            args.prompt,
            args.max_output_tokens,
        )


if __name__ == "__main__":
    main()
