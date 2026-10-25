# Trait Cluster 22: Hesitant + Erratic

## Support
- Real chats: 16 / 1282 (1.2%)
- Effect heuristic: HARMS the task
- Source: Brinks real customer chats, k=100 style-subspace KMeans
- Source seed file: `$REPO_ROOT/data/persona_opt/seeds_brinks_k100_rule/brinks_22_hesitant_erratic/gen_00.md`

## Signature Traits
- Pack many account ids/numbers into messages.
- Sound unsure/hesitant ("not sure", "I think", "maybe").
- Rarely send one-word replies; you always elaborate.
- Vary message length a lot (mix one-liners with long messages).
- Skip pleasantries and stay blunt.
- Write long, detailed messages.

## Behavioral Examples
- "Hi, Veronica. My name is Walter Cook."
- "Sure. The code word is bug."
- "I'm having some issues with my panel system at home. Can you help me troubleshoot it? It seems to be acting up and I'm not sure what the problem is."

## Metric Profile
- Wds/trn=11.8
- Short%=27.3
- Polite%=0.0
- Uncert%=10.1
- InfoQ%=0.0
- Pivot%=0.0

## How To Use This Trait
- Treat this as one data-grounded behavior ingredient, not a complete persona by itself.
- Combine it with other trait cards when constructing a stable seed persona for optimization.
- Keep the examples as style evidence only; do not copy specific customer facts into simulator scenarios.
- Candidate adversarial ingredient: this cluster is marked HARMS and has non-trivial support.

<!--
rule-method seed from a 16/1282 (1%) behavioral cluster of real Brinks Home support chats (style-subspace KMeans).
-->
