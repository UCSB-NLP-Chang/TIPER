# Trait Cluster 46: Erratic + Declarative

## Support
- Real chats: 49 / 1282 (3.8%)
- Effect heuristic: HARMS the task
- Source: Brinks real customer chats, k=100 style-subspace KMeans
- Source seed file: `$REPO_ROOT/data/persona_opt/seeds_brinks_k100_rule/brinks_46_erratic_declarative/gen_00.md`

## Signature Traits
- Vary message length a lot (mix one-liners with long messages).
- Rarely ask questions; you mostly state things.
- Reveal information gradually, a little at a time.
- Stay very polite — frequent please/thanks/sorry.
- Ask clarifying questions when confused ("what do you mean?").

## Behavioral Examples
- "C o s o."
- "I don't know it."
- "Can I integrate the ABCB<redacted> camera with my existing smart home like my Alexa or Google Home?"
- "No. That was all. Thank you."

## Metric Profile
- Wds/trn=4.7
- Short%=61.2
- Polite%=14.8
- Uncert%=3.5
- InfoQ%=0.0
- Pivot%=0.4

## How To Use This Trait
- Treat this as one data-grounded behavior ingredient, not a complete persona by itself.
- Combine it with other trait cards when constructing a stable seed persona for optimization.
- Keep the examples as style evidence only; do not copy specific customer facts into simulator scenarios.
- Candidate adversarial ingredient: this cluster is marked HARMS and has non-trivial support.

<!--
rule-method seed from a 49/1282 (4%) behavioral cluster of real Brinks Home support chats (style-subspace KMeans).
-->
