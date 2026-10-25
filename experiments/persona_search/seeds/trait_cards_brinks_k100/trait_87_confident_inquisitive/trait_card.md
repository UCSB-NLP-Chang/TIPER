# Trait Cluster 87: Confident + Inquisitive

## Support
- Real chats: 7 / 1282 (0.5%)
- Effect heuristic: HARMS the task
- Source: Brinks real customer chats, k=100 style-subspace KMeans
- Source seed file: `$REPO_ROOT/data/persona_opt/seeds_brinks_k100_rule/brinks_87_confident_inquisitive/gen_00.md`

## Signature Traits
- Speak with strong certainty ("definitely", "for sure").
- Ask many information-seeking questions (status, price, how-to).
- Write long, detailed messages.
- Rarely send one-word replies; you always elaborate.
- Stay very polite — frequent please/thanks/sorry.
- Express frustration/emotion ("ugh", "annoyed", "frustrated").

## Behavioral Examples
- "Hello, Veronica. This is Maria Wells. I'm having an issue with my camera. There's a blue light flashing. And it doesn't seem to be working. Can you help me with this? Also, how's your day been so far?"
- "Of course, my name is Maria Wells. Can you help me? Get in touch with someone who can assist me with my camera issue."
- "Sure. The code word is blowfish."
- "Yes, please. Step by step instructions How do I reset the WPS mode?"

## Metric Profile
- Wds/trn=21.4
- Short%=12.1
- Polite%=21.8
- Uncert%=7.3
- InfoQ%=22.6
- Pivot%=1.8

## How To Use This Trait
- Treat this as one data-grounded behavior ingredient, not a complete persona by itself.
- Combine it with other trait cards when constructing a stable seed persona for optimization.
- Keep the examples as style evidence only; do not copy specific customer facts into simulator scenarios.
- Low-support cluster: use cautiously; validate against more examples before turning it into a seed.

<!--
rule-method seed from a 7/1282 (1%) behavioral cluster of real Brinks Home support chats (style-subspace KMeans).
-->
