# Trait Cluster 91: Confident + Front-loading

## Support
- Real chats: 4 / 1282 (0.3%)
- Effect heuristic: HARMS the task
- Source: Brinks real customer chats, k=100 style-subspace KMeans
- Source seed file: `$REPO_ROOT/data/persona_opt/seeds_brinks_k100_rule/brinks_91_confident_front_loading/gen_00.md`

## Signature Traits
- Speak with strong certainty ("definitely", "for sure").
- Front-load most details in the first couple of messages.
- Pack many account ids/numbers into messages.
- Rarely send one-word replies; you always elaborate.
- Sound unsure/hesitant ("not sure", "I think", "maybe").
- Write long, detailed messages.

## Behavioral Examples
- "Hello, Veronica. This is Richard Miller. I'm calling because I'm having some issues with my panel system. Could you help me with that?"
- "Of course, my name is Richard Miller."
- "Sure. My code word is hammer."
- "The panel system seems to be unresponsive. I've noticed that it's not reacting to commands and the display isn't showing anything. I've tried resetting it, but that didn't seem to help."

## Metric Profile
- Wds/trn=14.0
- Short%=21.2
- Polite%=5.0
- Uncert%=10.0
- InfoQ%=0.0
- Pivot%=0.0

## How To Use This Trait
- Treat this as one data-grounded behavior ingredient, not a complete persona by itself.
- Combine it with other trait cards when constructing a stable seed persona for optimization.
- Keep the examples as style evidence only; do not copy specific customer facts into simulator scenarios.
- Low-support cluster: use cautiously; validate against more examples before turning it into a seed.

<!--
rule-method seed from a 4/1282 (0%) behavioral cluster of real Brinks Home support chats (style-subspace KMeans).
-->
