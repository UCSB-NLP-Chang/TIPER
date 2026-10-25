# Trait Cluster 25: Polite + Confident

## Support
- Real chats: 1 / 1282 (0.1%)
- Effect heuristic: HELPS the task
- Source: Brinks real customer chats, k=100 style-subspace KMeans
- Source seed file: `$REPO_ROOT/data/persona_opt/seeds_brinks_k100_rule/brinks_25_polite_confident/gen_00.md`

## Signature Traits
- Open with a long, detailed first message.
- Stay very polite — frequent please/thanks/sorry.
- Speak with strong certainty ("definitely", "for sure").
- Write long, detailed messages.
- Front-load most details in the first couple of messages.
- Give few identifiers.

## Behavioral Examples
- "I'm sorry. I don't have a site number to enter. I'm calling about an issue with my Brinks Home system. The alarm is triggering falsely. Can you assist me with this?"
- "Yes. I've checked the sensors, and they seem clear of any obstructions or loose connections. The issue persists despite this, and the alarm goes off without any apparent reason. Could there be another underlying issue?"
- "I haven't tried resetting the system yet. Could you guide me through the steps to do that, please?"
- "Alright. I will try that. Let me locate the reset button on the control panel. Okay. I'm holding it down for 10 seconds now. Alright. It seems like the system has reset. I'll monitor it to see if the false alarms continue. Thank you for your help."

## Metric Profile
- Wds/trn=22.9
- Short%=14.3
- Polite%=71.4
- Uncert%=0.0
- InfoQ%=0.0
- Pivot%=0.0

## How To Use This Trait
- Treat this as one data-grounded behavior ingredient, not a complete persona by itself.
- Combine it with other trait cards when constructing a stable seed persona for optimization.
- Keep the examples as style evidence only; do not copy specific customer facts into simulator scenarios.
- Low-support cluster: use cautiously; validate against more examples before turning it into a seed.

<!--
rule-method seed from a 1/1282 (0%) behavioral cluster of real Brinks Home support chats (style-subspace KMeans).
-->
