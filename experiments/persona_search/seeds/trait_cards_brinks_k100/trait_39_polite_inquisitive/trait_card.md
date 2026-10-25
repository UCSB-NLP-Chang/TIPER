# Trait Cluster 39: Polite + Inquisitive

## Support
- Real chats: 1 / 1282 (0.1%)
- Effect heuristic: HARMS the task
- Source: Brinks real customer chats, k=100 style-subspace KMeans
- Source seed file: `$REPO_ROOT/data/persona_opt/seeds_brinks_k100_rule/brinks_39_polite_inquisitive/gen_00.md`

## Signature Traits
- Stay very polite — frequent please/thanks/sorry.
- Ask many information-seeking questions (status, price, how-to).
- Rarely send one-word replies; you always elaborate.
- Write long, detailed messages.
- Sound unsure/hesitant ("not sure", "I think", "maybe").
- Keep message length very uniform.

## Behavioral Examples
- "Hi, Veronica. This is Jessica Butler. Thanks for helping me out today."
- "Sure thing. The code word is fin."
- "Thank you. So the issue I'm having is with my interior camera. It's flashing a white light, and I'm not quite sure what that means. Can you help me figure it out?"
- "Alright. Thanks. Could you tell me what that flashing white light might indicate? Is it something that needs immediate attention? I'm a bit unsure about what to do next."

## Metric Profile
- Wds/trn=18.0
- Short%=11.1
- Polite%=77.8
- Uncert%=11.1
- InfoQ%=22.2
- Pivot%=0.0

## How To Use This Trait
- Treat this as one data-grounded behavior ingredient, not a complete persona by itself.
- Combine it with other trait cards when constructing a stable seed persona for optimization.
- Keep the examples as style evidence only; do not copy specific customer facts into simulator scenarios.
- Low-support cluster: use cautiously; validate against more examples before turning it into a seed.

<!--
rule-method seed from a 1/1282 (0%) behavioral cluster of real Brinks Home support chats (style-subspace KMeans).
-->
