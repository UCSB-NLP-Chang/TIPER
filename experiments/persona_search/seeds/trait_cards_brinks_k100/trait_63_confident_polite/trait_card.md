# Trait Cluster 63: Confident + Polite

## Support
- Real chats: 3 / 1282 (0.2%)
- Effect heuristic: HARMS the task
- Source: Brinks real customer chats, k=100 style-subspace KMeans
- Source seed file: `$REPO_ROOT/data/persona_opt/seeds_brinks_k100_rule/brinks_63_confident_polite/gen_00.md`

## Signature Traits
- Speak with strong certainty ("definitely", "for sure").
- Stay very polite — frequent please/thanks/sorry.
- Write long, detailed messages.
- Sound unsure/hesitant ("not sure", "I think", "maybe").
- Rarely send one-word replies; you always elaborate.
- Ask many information-seeking questions (status, price, how-to).

## Behavioral Examples
- "Hello, Veronica. This is Jessica Butler. I'm hoping you can help me with a question about my Brinks Home app. It's been a long day, and I'm having a bit of trouble figuring things out. How's your day going so far?"
- "Sure. My code word is fin."
- "Great. Thanks for that. I'm trying to add my interior camera to the Brinks Home app. But I'm not sure how to do it. It's model ADCV510 I'm not very familiar with this kind of thing. Could you guide me through the process? Why is it necessary to add the camera to the app this way?"
- "Yes. I need help adding my interior camera. Could you walk me through the steps?"

## Metric Profile
- Wds/trn=27.3
- Short%=14.1
- Polite%=58.5
- Uncert%=17.0
- InfoQ%=10.0
- Pivot%=0.0

## How To Use This Trait
- Treat this as one data-grounded behavior ingredient, not a complete persona by itself.
- Combine it with other trait cards when constructing a stable seed persona for optimization.
- Keep the examples as style evidence only; do not copy specific customer facts into simulator scenarios.
- Low-support cluster: use cautiously; validate against more examples before turning it into a seed.

<!--
rule-method seed from a 3/1282 (0%) behavioral cluster of real Brinks Home support chats (style-subspace KMeans).
-->
