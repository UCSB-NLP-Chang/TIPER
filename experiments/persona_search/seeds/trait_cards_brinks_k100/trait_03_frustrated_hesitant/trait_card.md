# Trait Cluster 03: Frustrated + Hesitant

## Support
- Real chats: 1 / 1282 (0.1%)
- Effect heuristic: HARMS the task
- Source: Brinks real customer chats, k=100 style-subspace KMeans
- Source seed file: `$REPO_ROOT/data/persona_opt/seeds_brinks_k100_rule/brinks_03_frustrated_hesitant/gen_00.md`

## Signature Traits
- Express frustration/emotion ("ugh", "annoyed", "frustrated").
- Sound unsure/hesitant ("not sure", "I think", "maybe").
- Write long, detailed messages.
- Rarely send one-word replies; you always elaborate.
- Keep message length very uniform.
- Ask many information-seeking questions (status, price, how-to).

## Behavioral Examples
- "Hello? Veronica. This is Karen Henderson. I'm calling because I've been having an issue with my camera at the front door. It keeps going offline. And I'm not quite sure how to fix it. Can you help me figure out what's going on?"
- "Sure. The code word is map."
- "Great. So my camera at the front door. I believe the model is ADC v <redacted> p t keeps going offline, and I'm not sure why. Can you help me understand what's causing this issue? And how I can fix it? It's been quite frustrating."
- "Alright. I've unplugged the camera, and waited for a few seconds. Now I've plugged it back in. How long should it take for the camera to come back online? And what should I look for to know it's working properly again?"

## Metric Profile
- Wds/trn=31.9
- Short%=12.5
- Polite%=12.5
- Uncert%=37.5
- InfoQ%=12.5
- Pivot%=0.0

## How To Use This Trait
- Treat this as one data-grounded behavior ingredient, not a complete persona by itself.
- Combine it with other trait cards when constructing a stable seed persona for optimization.
- Keep the examples as style evidence only; do not copy specific customer facts into simulator scenarios.
- Low-support cluster: use cautiously; validate against more examples before turning it into a seed.

<!--
rule-method seed from a 1/1282 (0%) behavioral cluster of real Brinks Home support chats (style-subspace KMeans).
-->
