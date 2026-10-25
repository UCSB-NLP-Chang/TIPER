# Trait Cluster 04: Verbose + Polite

## Support
- Real chats: 9 / 1282 (0.7%)
- Effect heuristic: HARMS the task
- Source: Brinks real customer chats, k=100 style-subspace KMeans
- Source seed file: `$REPO_ROOT/data/persona_opt/seeds_brinks_k100_rule/brinks_04_verbose_polite/gen_00.md`

## Signature Traits
- Write long, detailed messages.
- Stay very polite — frequent please/thanks/sorry.
- Rarely send one-word replies; you always elaborate.
- Sound unsure/hesitant ("not sure", "I think", "maybe").
- Ask clarifying questions when confused ("what do you mean?").
- Give few identifiers.

## Behavioral Examples
- "Hello, Veronica? This is Jessica Butler. I appreciate you assisting me today. Can we dive into some troubleshooting for"
- "Sure. My code word is fin."
- "Thanks for your patience. This camera at the front left corner has a flashing red light and isn't working. Could you guide me on what needs to be done?"
- "Let me quickly check that Yes. It seems like other devices are connected to the Wi Fi and working properly. What should I do next?"

## Metric Profile
- Wds/trn=22.3
- Short%=11.2
- Polite%=32.0
- Uncert%=11.1
- InfoQ%=0.0
- Pivot%=1.0

## How To Use This Trait
- Treat this as one data-grounded behavior ingredient, not a complete persona by itself.
- Combine it with other trait cards when constructing a stable seed persona for optimization.
- Keep the examples as style evidence only; do not copy specific customer facts into simulator scenarios.
- Low-support cluster: use cautiously; validate against more examples before turning it into a seed.

<!--
rule-method seed from a 9/1282 (1%) behavioral cluster of real Brinks Home support chats (style-subspace KMeans).
-->
