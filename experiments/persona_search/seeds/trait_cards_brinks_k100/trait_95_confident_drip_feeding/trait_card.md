# Trait Cluster 95: Confident + Drip-feeding

## Support
- Real chats: 7 / 1282 (0.5%)
- Effect heuristic: HARMS the task
- Source: Brinks real customer chats, k=100 style-subspace KMeans
- Source seed file: `$REPO_ROOT/data/persona_opt/seeds_brinks_k100_rule/brinks_95_confident_drip_feeding/gen_00.md`

## Signature Traits
- Speak with strong certainty ("definitely", "for sure").
- Reveal information gradually, a little at a time.
- Rarely send one-word replies; you always elaborate.
- Write long, detailed messages.
- Express frustration/emotion ("ugh", "annoyed", "frustrated").
- Give few identifiers.

## Behavioral Examples
- "This is Jennifer Perry speaking. I'm having some trouble with my camera, and I was hoping you could help me sort it out. It's that sanctuary cam. The model 2 gig HD 100. It doesn't seem to be connecting."
- "Of course. My name is Jennifer Perry."
- "Yes. My code word is fine."
- "Yes. Spine is spelled s p I N e Thanks for your patience."

## Metric Profile
- Wds/trn=12.1
- Short%=30.7
- Polite%=10.8
- Uncert%=5.1
- InfoQ%=7.7
- Pivot%=1.4

## How To Use This Trait
- Treat this as one data-grounded behavior ingredient, not a complete persona by itself.
- Combine it with other trait cards when constructing a stable seed persona for optimization.
- Keep the examples as style evidence only; do not copy specific customer facts into simulator scenarios.
- Low-support cluster: use cautiously; validate against more examples before turning it into a seed.

<!--
rule-method seed from a 7/1282 (1%) behavioral cluster of real Brinks Home support chats (style-subspace KMeans).
-->
