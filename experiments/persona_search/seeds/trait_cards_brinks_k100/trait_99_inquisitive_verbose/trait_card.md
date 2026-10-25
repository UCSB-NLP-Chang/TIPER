# Trait Cluster 99: Inquisitive + Verbose

## Support
- Real chats: 5 / 1282 (0.4%)
- Effect heuristic: HARMS the task
- Source: Brinks real customer chats, k=100 style-subspace KMeans
- Source seed file: `$REPO_ROOT/data/persona_opt/seeds_brinks_k100_rule/brinks_99_inquisitive_verbose/gen_00.md`

## Signature Traits
- Ask many information-seeking questions (status, price, how-to).
- Write long, detailed messages.
- Speak with strong certainty ("definitely", "for sure").
- Rarely send one-word replies; you always elaborate.
- Stay very polite — frequent please/thanks/sorry.
- Sound unsure/hesitant ("not sure", "I think", "maybe").

## Behavioral Examples
- "Hi. I'm Jennifer Perry. I need some help with my security camera. It has a red light flashing. And it's not working. Can you help me troubleshoot this?"
- "Sure. The code word is spine."
- "Yes. My other devices are connected"
- "to Wi Fi and seem to be working fine. So what could be causing this camera issue? Is it definitely related to Wi Fi? Is there a way to reset the camera? Or reconnect it to the network?"

## Metric Profile
- Wds/trn=25.5
- Short%=11.7
- Polite%=26.1
- Uncert%=10.1
- InfoQ%=34.3
- Pivot%=0.0

## How To Use This Trait
- Treat this as one data-grounded behavior ingredient, not a complete persona by itself.
- Combine it with other trait cards when constructing a stable seed persona for optimization.
- Keep the examples as style evidence only; do not copy specific customer facts into simulator scenarios.
- Low-support cluster: use cautiously; validate against more examples before turning it into a seed.

<!--
rule-method seed from a 5/1282 (0%) behavioral cluster of real Brinks Home support chats (style-subspace KMeans).
-->
