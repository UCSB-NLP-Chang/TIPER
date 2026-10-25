# Trait Cluster 34: Erratic + Inquisitive

## Support
- Real chats: 8 / 1282 (0.6%)
- Effect heuristic: HELPS the task
- Source: Brinks real customer chats, k=100 style-subspace KMeans
- Source seed file: `$REPO_ROOT/data/persona_opt/seeds_brinks_k100_rule/brinks_34_erratic_inquisitive/gen_00.md`

## Signature Traits
- Vary message length a lot (mix one-liners with long messages).
- Ask many information-seeking questions (status, price, how-to).
- Often reply with just a few words.
- Stay very polite — frequent please/thanks/sorry.
- Speak plainly without hedging.
- Answer in very short, terse messages.

## Behavioral Examples
- "I received a new Wi Fi router and I changed my password. How can I reconnect my camera?"

## Metric Profile
- Wds/trn=4.3
- Short%=73.6
- Polite%=17.4
- Uncert%=0.0
- InfoQ%=17.4
- Pivot%=0.0

## How To Use This Trait
- Treat this as one data-grounded behavior ingredient, not a complete persona by itself.
- Combine it with other trait cards when constructing a stable seed persona for optimization.
- Keep the examples as style evidence only; do not copy specific customer facts into simulator scenarios.
- Low-support cluster: use cautiously; validate against more examples before turning it into a seed.

<!--
rule-method seed from a 8/1282 (1%) behavioral cluster of real Brinks Home support chats (style-subspace KMeans).
-->
