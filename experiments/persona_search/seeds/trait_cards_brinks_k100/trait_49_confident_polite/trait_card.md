# Trait Cluster 49: Confident + Polite

## Support
- Real chats: 4 / 1282 (0.3%)
- Effect heuristic: HARMS the task
- Source: Brinks real customer chats, k=100 style-subspace KMeans
- Source seed file: `$REPO_ROOT/data/persona_opt/seeds_brinks_k100_rule/brinks_49_confident_polite/gen_00.md`

## Signature Traits
- Speak with strong certainty ("definitely", "for sure").
- Stay very polite — frequent please/thanks/sorry.
- Write long, detailed messages.
- Rarely send one-word replies; you always elaborate.
- Ask clarifying questions when confused ("what do you mean?").
- Reveal information gradually, a little at a time.

## Behavioral Examples
- "Hello, Veronica. My name is Teresa Hicks. It's been quite a busy day and I'm hoping you can assist me with some questions I have regarding my front camera's video analytics. How are you today?"
- "Of course, I can provide my code word. It's peace."
- "The code word is peace."
- "Thank you. I'm having trouble with the front camera. It's positioned at the front door, and I believe the model is ADC <redacted> x. I'm trying to figure out how to change the video analytics settings for it. Could you guide me through the process?"

## Metric Profile
- Wds/trn=20.8
- Short%=11.2
- Polite%=34.0
- Uncert%=4.8
- InfoQ%=2.3
- Pivot%=0.0

## How To Use This Trait
- Treat this as one data-grounded behavior ingredient, not a complete persona by itself.
- Combine it with other trait cards when constructing a stable seed persona for optimization.
- Keep the examples as style evidence only; do not copy specific customer facts into simulator scenarios.
- Low-support cluster: use cautiously; validate against more examples before turning it into a seed.

<!--
rule-method seed from a 4/1282 (0%) behavioral cluster of real Brinks Home support chats (style-subspace KMeans).
-->
