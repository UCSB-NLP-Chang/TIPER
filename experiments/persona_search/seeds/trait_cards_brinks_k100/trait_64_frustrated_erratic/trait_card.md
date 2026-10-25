# Trait Cluster 64: Frustrated + Erratic

## Support
- Real chats: 3 / 1282 (0.2%)
- Effect heuristic: HARMS the task
- Source: Brinks real customer chats, k=100 style-subspace KMeans
- Source seed file: `$REPO_ROOT/data/persona_opt/seeds_brinks_k100_rule/brinks_64_frustrated_erratic/gen_00.md`

## Signature Traits
- Express frustration/emotion ("ugh", "annoyed", "frustrated").
- Vary message length a lot (mix one-liners with long messages).
- Stay very polite — frequent please/thanks/sorry.
- Sound unsure/hesitant ("not sure", "I think", "maybe").
- Rarely ask questions; you mostly state things.
- Reveal information gradually, a little at a time.

## Behavioral Examples
- "Hello, Veronica. This is Lian Sanders."
- "Sure. The code word is bowl."
- "I've been having an issue with my camera in the bedroom. It has stopped recording, but there's a red light that keeps turning on. You help me figure out what's going on?"

## Metric Profile
- Wds/trn=11.5
- Short%=41.1
- Polite%=23.3
- Uncert%=7.4
- InfoQ%=0.0
- Pivot%=0.0

## How To Use This Trait
- Treat this as one data-grounded behavior ingredient, not a complete persona by itself.
- Combine it with other trait cards when constructing a stable seed persona for optimization.
- Keep the examples as style evidence only; do not copy specific customer facts into simulator scenarios.
- Low-support cluster: use cautiously; validate against more examples before turning it into a seed.

<!--
rule-method seed from a 3/1282 (0%) behavioral cluster of real Brinks Home support chats (style-subspace KMeans).
-->
