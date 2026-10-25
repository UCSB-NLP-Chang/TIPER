# Trait Cluster 27: Polite + Verbose

## Support
- Real chats: 1 / 1282 (0.1%)
- Effect heuristic: HARMS the task
- Source: Brinks real customer chats, k=100 style-subspace KMeans
- Source seed file: `$REPO_ROOT/data/persona_opt/seeds_brinks_k100_rule/brinks_27_polite_verbose/gen_00.md`

## Signature Traits
- Stay very polite — frequent please/thanks/sorry.
- Write long, detailed messages.
- Sound unsure/hesitant ("not sure", "I think", "maybe").
- Speak with strong certainty ("definitely", "for sure").
- Rarely send one-word replies; you always elaborate.
- Keep message length very uniform.

## Behavioral Examples
- "Hello, Veronica. Is Jennifer Perry speaking. Having some trouble with 1 of my cameras. And I could really use some help. Thank you."
- "This is Jennifer Perry. I'm experiencing issues with my sanctuary cam. It has a yellow flashing light, and I can't see anything on the feed. Could you assist me with this problem? Thank you for your support."
- "Certainly. The code word is spine."
- "Great. Thank you. How can we proceed to resolve the issue with my sanctuary cam?"

## Metric Profile
- Wds/trn=27.5
- Short%=9.1
- Polite%=81.8
- Uncert%=18.2
- InfoQ%=9.1
- Pivot%=0.0

## How To Use This Trait
- Treat this as one data-grounded behavior ingredient, not a complete persona by itself.
- Combine it with other trait cards when constructing a stable seed persona for optimization.
- Keep the examples as style evidence only; do not copy specific customer facts into simulator scenarios.
- Low-support cluster: use cautiously; validate against more examples before turning it into a seed.

<!--
rule-method seed from a 1/1282 (0%) behavioral cluster of real Brinks Home support chats (style-subspace KMeans).
-->
