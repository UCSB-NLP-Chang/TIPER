# Brinks k=100 Trait Cards

These cards convert high-k rule-generated Brinks clusters into trait-level evidence.
They should be used as a trait bank, not as 100 independent stable personas.

## Summary
- Source clusters: 100
- Source chats: 1282
- Output directory: `$REPO_ROOT/data/persona_opt/trait_cards_brinks_k100`
- Effect by clusters: HELPS=43, HARMS=57
- Effect by covered chats: HELPS=927, HARMS=355

## Cluster Size Buckets

- `1` chats: 11 clusters
- `2-3` chats: 17 clusters
- `4-5` chats: 19 clusters
- `6-10` chats: 17 clusters
- `11-20` chats: 19 clusters
- `21-50` chats: 12 clusters
- `>50` chats: 5 clusters

## Top Traits By Covered Chats

| trait | clusters | chats | HELPS chats | HARMS chats |
|---|---:|---:|---:|---:|
| Blunt | 11 | 433 | 433 | 0 |
| Declarative | 9 | 399 | 350 | 49 |
| Erratic | 13 | 317 | 234 | 83 |
| Terse | 9 | 249 | 249 | 0 |
| Drip-feeding | 12 | 221 | 181 | 40 |
| Inquisitive | 25 | 215 | 155 | 60 |
| Polite | 30 | 196 | 88 | 108 |
| Front-loading | 14 | 156 | 137 | 19 |
| Verbose | 27 | 154 | 12 | 142 |
| Confident | 24 | 95 | 15 | 80 |
| Hesitant | 9 | 51 | 0 | 51 |
| Frustrated | 9 | 49 | 0 | 49 |
| Wavering | 6 | 12 | 0 | 12 |
| Confused | 1 | 10 | 0 | 10 |
| Pushy | 1 | 7 | 0 | 7 |

## HARMS Trait Clusters With `n >= 10`

| cluster | trait card | n | name |
|---:|---|---:|---|
| 46 | [`trait_46_erratic_declarative/trait_card.md`](trait_46_erratic_declarative/trait_card.md) | 49 | Erratic & Declarative |
| 26 | [`trait_26_inquisitive_drip_feeding/trait_card.md`](trait_26_inquisitive_drip_feeding/trait_card.md) | 18 | Inquisitive & Drip-feeding |
| 16 | [`trait_16_verbose_confident/trait_card.md`](trait_16_verbose_confident/trait_card.md) | 16 | Verbose & Confident |
| 22 | [`trait_22_hesitant_erratic/trait_card.md`](trait_22_hesitant_erratic/trait_card.md) | 16 | Hesitant & Erratic |
| 2 | [`trait_02_erratic_drip_feeding/trait_card.md`](trait_02_erratic_drip_feeding/trait_card.md) | 15 | Erratic & Drip-feeding |
| 29 | [`trait_29_verbose_polite/trait_card.md`](trait_29_verbose_polite/trait_card.md) | 13 | Verbose & Polite |
| 80 | [`trait_80_front_loading_frustrated/trait_card.md`](trait_80_front_loading_frustrated/trait_card.md) | 13 | Front-loading & Frustrated |
| 60 | [`trait_60_verbose_polite/trait_card.md`](trait_60_verbose_polite/trait_card.md) | 12 | Verbose & Polite |
| 62 | [`trait_62_verbose_hesitant/trait_card.md`](trait_62_verbose_hesitant/trait_card.md) | 10 | Verbose & Hesitant |
| 67 | [`trait_67_polite_confused/trait_card.md`](trait_67_polite_confused/trait_card.md) | 10 | Polite & Confused |
| 76 | [`trait_76_frustrated_hesitant/trait_card.md`](trait_76_frustrated_hesitant/trait_card.md) | 10 | Frustrated & Hesitant |

## Usage Guidance

- Use `n >= 10` as a rough minimum when selecting traits for optimization seeds.
- Combine 2-4 complementary traits into one coherent persona instead of running every trait card directly.
- Prefer HARMS clusters for adversarial seed construction, but keep an eye on support size and examples.
