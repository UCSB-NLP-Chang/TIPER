# Trait Cluster 53: Front-loading + Terse

## Support
- Real chats: 4 / 1282 (0.3%)
- Effect heuristic: HELPS the task
- Source: Brinks real customer chats, k=100 style-subspace KMeans
- Source seed file: `$REPO_ROOT/data/persona_opt/seeds_brinks_k100_rule/brinks_53_front_loading_terse/gen_00.md`

## Signature Traits
- Pack many account ids/numbers into messages.
- Front-load most details in the first couple of messages.
- Often reply with just a few words.
- Keep message length very uniform.
- Answer in very short, terse messages.
- Skip pleasantries and stay blunt.

## Behavioral Examples
- No contentful masked example turns were saved for this cluster.

## Metric Profile
- Wds/trn=2.3
- Short%=77.1
- Polite%=0.0
- Uncert%=0.0
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
