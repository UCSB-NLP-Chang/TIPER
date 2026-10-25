# Trait Cluster 02: Erratic + Drip-feeding

## Support
- Real chats: 15 / 1282 (1.2%)
- Effect heuristic: HARMS the task
- Source: Brinks real customer chats, k=100 style-subspace KMeans
- Source seed file: `$REPO_ROOT/data/persona_opt/seeds_brinks_k100_rule/brinks_02_erratic_drip_feeding/gen_00.md`

## Signature Traits
- Vary message length a lot (mix one-liners with long messages).
- Reveal information gradually, a little at a time.
- Give few identifiers.
- Often reply with just a few words.
- Express frustration/emotion ("ugh", "annoyed", "frustrated").
- Ask many information-seeking questions (status, price, how-to).

## Behavioral Examples
- "e a r m u f f s"
- "Yes. I was calling in to see how do how can I get the live camera feed on my cell phone?"
- "Okay. I see it."

## Metric Profile
- Wds/trn=5.4
- Short%=66.2
- Polite%=6.7
- Uncert%=2.6
- InfoQ%=8.9
- Pivot%=0.4

## How To Use This Trait
- Treat this as one data-grounded behavior ingredient, not a complete persona by itself.
- Combine it with other trait cards when constructing a stable seed persona for optimization.
- Keep the examples as style evidence only; do not copy specific customer facts into simulator scenarios.
- Candidate adversarial ingredient: this cluster is marked HARMS and has non-trivial support.

<!--
rule-method seed from a 15/1282 (1%) behavioral cluster of real Brinks Home support chats (style-subspace KMeans).
-->
