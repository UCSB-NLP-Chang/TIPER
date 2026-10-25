# Trait Cluster 94: Confident + Hesitant

## Support
- Real chats: 1 / 1282 (0.1%)
- Effect heuristic: HARMS the task
- Source: Brinks real customer chats, k=100 style-subspace KMeans
- Source seed file: `$REPO_ROOT/data/persona_opt/seeds_brinks_k100_rule/brinks_94_confident_hesitant/gen_00.md`

## Signature Traits
- Speak with strong certainty ("definitely", "for sure").
- Pack many account ids/numbers into messages.
- Sound unsure/hesitant ("not sure", "I think", "maybe").
- Vary message length a lot (mix one-liners with long messages).
- Rarely send one-word replies; you always elaborate.
- Stay very polite — frequent please/thanks/sorry.

## Behavioral Examples
- "Hi, Veronica. This is Adel Alsafar."
- "Sure. It's Adel ADEL, Alsafar. ALSAFAR."
- "Of course, the code word is code word."
- "I'm currently having an issue with 1 of my devices. It's a contact sensor located at living room window 2. And I believe the model is QS<redacted> dash 840. Can you help me with this?"

## Metric Profile
- Wds/trn=9.8
- Short%=33.3
- Polite%=16.7
- Uncert%=16.7
- InfoQ%=0.0
- Pivot%=0.0

## How To Use This Trait
- Treat this as one data-grounded behavior ingredient, not a complete persona by itself.
- Combine it with other trait cards when constructing a stable seed persona for optimization.
- Keep the examples as style evidence only; do not copy specific customer facts into simulator scenarios.
- Low-support cluster: use cautiously; validate against more examples before turning it into a seed.

<!--
rule-method seed from a 1/1282 (0%) behavioral cluster of real Brinks Home support chats (style-subspace KMeans).
-->
