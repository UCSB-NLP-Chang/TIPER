# Trait Cluster 37: Front-loading + Terse

## Support
- Real chats: 24 / 1282 (1.9%)
- Effect heuristic: HELPS the task
- Source: Brinks real customer chats, k=100 style-subspace KMeans
- Source seed file: `$REPO_ROOT/data/persona_opt/seeds_brinks_k100_rule/brinks_37_front_loading_terse/gen_00.md`

## Signature Traits
- Front-load most details in the first couple of messages.
- Keep message length very uniform.
- Often reply with just a few words.
- Pack many account ids/numbers into messages.
- Answer in very short, terse messages.
- Skip pleasantries and stay blunt.

## Behavioral Examples
- No contentful masked example turns were saved for this cluster.

## Metric Profile
- Wds/trn=1.8
- Short%=92.7
- Polite%=0.0
- Uncert%=1.4
- InfoQ%=0.0
- Pivot%=0.0

## How To Use This Trait
- Treat this as one data-grounded behavior ingredient, not a complete persona by itself.
- Combine it with other trait cards when constructing a stable seed persona for optimization.
- Keep the examples as style evidence only; do not copy specific customer facts into simulator scenarios.

<!--
rule-method seed from a 24/1282 (2%) behavioral cluster of real Brinks Home support chats (style-subspace KMeans).
-->
