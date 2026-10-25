# Trait Cluster 57: Confident + Verbose

## Support
- Real chats: 4 / 1282 (0.3%)
- Effect heuristic: HARMS the task
- Source: Brinks real customer chats, k=100 style-subspace KMeans
- Source seed file: `$REPO_ROOT/data/persona_opt/seeds_brinks_k100_rule/brinks_57_confident_verbose/gen_00.md`

## Signature Traits
- Speak with strong certainty ("definitely", "for sure").
- Write long, detailed messages.
- Change your mind or switch topic mid-conversation ("actually...", "never mind").
- Rarely send one-word replies; you always elaborate.
- Skip pleasantries and stay blunt.

## Behavioral Examples
- "Hello, Veronica. It's Maria Wells here. How's your day going so far?"
- "Certainly, Veronica. The code word is blowfish."
- "Well, I've been having an issue with 1 of my cameras. I've reset it multiple times, but it still remains offline. Could you help me figure out why this is happening? It's located at the SVR."

## Metric Profile
- Wds/trn=18.2
- Short%=21.9
- Polite%=0.0
- Uncert%=3.1
- InfoQ%=3.1
- Pivot%=3.1

## How To Use This Trait
- Treat this as one data-grounded behavior ingredient, not a complete persona by itself.
- Combine it with other trait cards when constructing a stable seed persona for optimization.
- Keep the examples as style evidence only; do not copy specific customer facts into simulator scenarios.
- Low-support cluster: use cautiously; validate against more examples before turning it into a seed.

<!--
rule-method seed from a 4/1282 (0%) behavioral cluster of real Brinks Home support chats (style-subspace KMeans).
-->
