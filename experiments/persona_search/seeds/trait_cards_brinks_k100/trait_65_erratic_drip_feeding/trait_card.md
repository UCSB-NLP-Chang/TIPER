# Trait Cluster 65: Erratic + Drip-feeding

## Support
- Real chats: 6 / 1282 (0.5%)
- Effect heuristic: HELPS the task
- Source: Brinks real customer chats, k=100 style-subspace KMeans
- Source seed file: `$REPO_ROOT/data/persona_opt/seeds_brinks_k100_rule/brinks_65_erratic_drip_feeding/gen_00.md`

## Signature Traits
- Vary message length a lot (mix one-liners with long messages).
- Reveal information gradually, a little at a time.
- Often reply with bare acknowledgements (ok, yes, thanks).
- Give few identifiers.
- Often reply with just a few words.

## Behavioral Examples
- "Yeah. The light on the camera isn't flashing blue for a while. Can you tell me what's going on and what we know what I need to do to fix this?"
- "Okay. What's the next step?"
- "Okay. I've done that."

## Metric Profile
- Wds/trn=8.5
- Short%=64.8
- Polite%=3.5
- Uncert%=2.4
- InfoQ%=7.7
- Pivot%=0.0

## How To Use This Trait
- Treat this as one data-grounded behavior ingredient, not a complete persona by itself.
- Combine it with other trait cards when constructing a stable seed persona for optimization.
- Keep the examples as style evidence only; do not copy specific customer facts into simulator scenarios.
- Low-support cluster: use cautiously; validate against more examples before turning it into a seed.

<!--
rule-method seed from a 6/1282 (0%) behavioral cluster of real Brinks Home support chats (style-subspace KMeans).
-->
