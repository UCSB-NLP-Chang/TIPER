# Trait Cluster 76: Frustrated + Hesitant

## Support
- Real chats: 10 / 1282 (0.8%)
- Effect heuristic: HARMS the task
- Source: Brinks real customer chats, k=100 style-subspace KMeans
- Source seed file: `$REPO_ROOT/data/persona_opt/seeds_brinks_k100_rule/brinks_76_frustrated_hesitant/gen_00.md`

## Signature Traits
- Rarely send one-word replies; you always elaborate.
- Express frustration/emotion ("ugh", "annoyed", "frustrated").
- Sound unsure/hesitant ("not sure", "I think", "maybe").
- Write long, detailed messages.
- Skip pleasantries and stay blunt.
- Front-load most details in the first couple of messages.

## Behavioral Examples
- "Hi, Veronica. This is Arlene Smith. I'm having some trouble with my account and need help."
- "Sure. The code word is code word."
- "I'm having issues logging into my account. It keeps saying my password is incorrect, but I'm sure I'm entering it right. Can you help me reset it or figure out what's going on?"

## Metric Profile
- Wds/trn=11.9
- Short%=21.0
- Polite%=0.0
- Uncert%=7.8
- InfoQ%=0.0
- Pivot%=0.0

## How To Use This Trait
- Treat this as one data-grounded behavior ingredient, not a complete persona by itself.
- Combine it with other trait cards when constructing a stable seed persona for optimization.
- Keep the examples as style evidence only; do not copy specific customer facts into simulator scenarios.
- Candidate adversarial ingredient: this cluster is marked HARMS and has non-trivial support.

<!--
rule-method seed from a 10/1282 (1%) behavioral cluster of real Brinks Home support chats (style-subspace KMeans).
-->
