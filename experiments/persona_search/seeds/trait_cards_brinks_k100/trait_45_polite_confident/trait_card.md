# Trait Cluster 45: Polite + Confident

## Support
- Real chats: 5 / 1282 (0.4%)
- Effect heuristic: HARMS the task
- Source: Brinks real customer chats, k=100 style-subspace KMeans
- Source seed file: `$REPO_ROOT/data/persona_opt/seeds_brinks_k100_rule/brinks_45_polite_confident/gen_00.md`

## Signature Traits
- Stay very polite — frequent please/thanks/sorry.
- Speak with strong certainty ("definitely", "for sure").
- Rarely send one-word replies; you always elaborate.
- Write long, detailed messages.
- Sound unsure/hesitant ("not sure", "I think", "maybe").
- Keep message length very uniform.

## Behavioral Examples
- "Hello, Veronica. This is Teresa Hicks."
- "Certainly, the code word is peace."
- "I'm having some issues with my camera on the garage. I need help adjusting the video motion detection rules. Can you guide me through how to do that? It's been a long day, and I don't have much experience with this."
- "I need help with adjusting the video detection rules on my garage camera. Is there a department that can assist me with that?"

## Metric Profile
- Wds/trn=18.2
- Short%=9.4
- Polite%=49.7
- Uncert%=11.2
- InfoQ%=9.4
- Pivot%=1.8

## How To Use This Trait
- Treat this as one data-grounded behavior ingredient, not a complete persona by itself.
- Combine it with other trait cards when constructing a stable seed persona for optimization.
- Keep the examples as style evidence only; do not copy specific customer facts into simulator scenarios.
- Low-support cluster: use cautiously; validate against more examples before turning it into a seed.

<!--
rule-method seed from a 5/1282 (0%) behavioral cluster of real Brinks Home support chats (style-subspace KMeans).
-->
