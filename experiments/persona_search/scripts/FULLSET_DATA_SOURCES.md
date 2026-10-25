# Fullset (train+val+test @ 3 trials) — 数据源与补全性

本 doc 记录 persona-diff **全集**(train+val+test,每 task 恰好 3 trials)的**所有数据源**、
每个 cell 的补跑量,以及**补全性审计**结论。

- 组装+补跑脚本:`scripts/persona_opt/fullset_build.py`
- 汇总脚本:`scripts/persona_opt/fullset_summary.py`(出 avg@3 / pass^3 / Δ)
- 启动 wrapper:`scripts/persona_opt/run_fullset.sh`(全域,日志 `/tmp/fullset_run.log`);
  telecom 第二轮:`scripts/persona_opt/run_fullset_telecom.sh`(opt/no_persona/p0003,日志 `/tmp/fullset_telecom.log`)
- 输出:`data/simulations/fullset/<domain>/<island>_<cond>/results.json`(单文件 JSON checkpoint)
- 最后更新:2026-07-14

## ✅ 完成状态(2026-07-14)

**全部 28 个 cell 补全并核验通过**:每个 split 的每个 task 恰好 3 个 canonical trial,reward 字段齐全、seed 纯净(无非 CANON seed)。
- 首轮 21 cell(airline 5 + retail 7 + banking 5 + telecom seed 4)。
- **2026-07-14 第二轮补跑**:telecom **opt 4 岛 + no_persona + p0003(seed/opt,xhigh)= 7 cell**,共 1082 slots,~2h40m 跑完、0 error。
- sim 数:airline 每 cell 150、retail 342、banking 291、telecom(含 opt/no_p)342、p0003 342。
- **banking `task_012`**:`cw_04_seed`/`cw_04_opt` 各有 1 条 `infrastructure_error`(reward=None)→ 删除后重跑补回(现为正常 `user_stop` 真值)。
- 汇总口径见文末;结论:opt 在 test/全集上基本不优于 seed(过拟合塌缩),telecom/p0003 也印证,详见 memory `[[persona-opt-unified-results]]`。

> **重要更正(2026-07-14 实测)**:telecom 的 GEPA run(`enriched_<isl>_g*`、`ev2v4_p0003_g*`)**确实序列化了 persona 指纹**(非全 None),所以 telecom 的 **opt train/val 可按指纹部分 salvage**(不同于 airline/retail/banking 的 opt)。这与本 doc 早前"opt 全不可 salvage"的判断相反 —— 那条只对 **非 telecom 域**成立。

## 🚨 重大纠错(2026-07-14,用户复核):telecom brinks 建错,待修

telecom **brinks 4 岛(seed×4 + opt×4 + no_persona)= 9 个 cell 建错了,双重错误**:
1. **user effort 应为 `xhigh` 不是 `high`**(旧认知反了);
2. 因此 **opt winner persona 也跟着从 high 目录选错**(winner 按 val_agg 选,high/xhigh 跑出的难度不同 → 选中不同 candidate)。

**权威真值 = `data/simulations/ev2v4_cert/`**(user **xhigh** / agent high,54 task×3t),
对应 doc `data/persona_opt/docs/UNIFIED_RESULTS.md:54-69` + `PERSONA_EXPERIMENT_DATA.md:181-190`。
正确 opt winner(cert test **diff = 1−avg@3**):

| 岛 | 正确 winner persona | ev2v4_cert cell | diff | gen |
|---|---|---|---|---|
| brinks_00 | `a7020ec9a7` | b00_top1 | **0.438** | g08 |
| brinks_02 | `c655bad793` | b02_top1 | **0.432** | g01 |
| brinks_03 | `47c61c2629` | b03_top1 | **0.290** | g11 |
| brinks_04 | `e7405d0aa9` | b04_top1 | **0.278** | g06 |
| p0003 | `5dd389d157` | p0003_best | **0.259** | g13 |

**fullset 旧 brinks 用了 high + 错 persona**(`322b27e381`/`afceefc01b`/`469b445118`/`79e7796905`),
diff 全塌到 0.22–0.27;**唯有 p0003 用对了**(xhigh + `5dd389d157`,diff 0.259 完美复现),证明配置对即可复现用户目标值。

### ✅ 已零补跑落地(2026-07-14,用户定「接受跨 effort,零补跑」)

**采纳方案**:opt 四岛直接采用 `ev2v4_cert/b0X_top1`(xhigh、persona 对、**test-only**);
seed / no_persona **保留现有 high 完整数据**;两者 effort 不一致,telecom 的 Δ(opt−seed) 为**跨 effort,仅供参考**。
落地方式 = 用 ev2v4_cert 产物覆盖 `fullset/telecom/brinks_0X_opt/results.json`(旧错误 high 数据备份为 `.bak_high_wrongpersona`),**未跑任何 tau2**。

落地后 test-split opt diff(1−avg@3)与用户目标对照:

| 岛 | opt persona | opt avg@3 | diff | 用户目标 |
|---|---|---|---|---|
| brinks_00 | `a7020ec9a7` | 0.562 | **0.438** | 0.468(⚠差 0.03,见下) |
| brinks_02 | `c655bad793` | 0.568 | **0.432** | 0.432 ✅ |
| brinks_03 | `47c61c2629` | 0.710 | **0.290** | 0.290 ✅ |
| brinks_04 | `e7405d0aa9` | 0.722 | **0.278** | 0.278 ✅ |
| p0003 | `5dd389d157` | 0.741 | **0.259** | 0.260 ✅ |

> **brinks_00 = 0.438 是所有 GEPA cert 里 b00 最难的 optimized persona**(`UNIFIED_RESULTS.md:58`)。
> 用户记的 0.468 无对应 cert winner —— 最接近的 0.463 是同行另一列(difficulty 非 test-diff),
> 0.469 是 `naturalness_opt_v2/b00`(naturalness 重优化支线,非 GEPA cert)。零补跑前提下 b00 只能取 0.438。

telecom 落地后 test-split Δ(opt−seed,**跨 effort**):b00 −0.099 / b02 −0.259 / b03 −0.099 / b04 −0.049;
opt−no_p 全负(−0.09~−0.25)。⚠️ seed=high、opt=xhigh,此 Δ 不严格可比,仅供参考。

### ✅ opt full(train/val)已补跑完成(2026-07-14,464 slot,`run_fullset_telecom_opt_xhigh.sh`,0 error)

四岛 opt 现均为 **train30/val30/test54 @3t 齐全**(xhigh、正确 persona)。test 全 salvage 自 ev2v4_cert,train/val 从 `ev2v4_<isl>_g*` xhigh 指纹命中部分 salvage 后补齐。**最终 seed(high) vs opt(xhigh) full 对比**(avg@3 / pass³,Δ=opt−seed):

| 岛 | split | seed | opt | Δavg | Δpass³ |
|---|---|---|---|---|---|
| brinks_00 | test | 0.660/0.481 | 0.562/0.352 | −0.099 | −0.130 |
| brinks_00 | ALL  | 0.746/0.570 | 0.561/0.351 | −0.184 | −0.219 |
| brinks_02 | test | 0.827/0.704 | 0.568/0.370 | −0.259 | −0.333 |
| brinks_02 | ALL  | 0.839/0.675 | 0.573/0.333 | −0.266 | −0.342 |
| brinks_03 | test | 0.809/0.593 | 0.710/0.593 | −0.099 | +0.000 |
| brinks_03 | ALL  | 0.839/0.675 | 0.734/0.561 | −0.105 | −0.114 |
| brinks_04 | test | 0.772/0.648 | 0.722/0.519 | −0.049 | −0.130 |
| brinks_04 | ALL  | 0.789/0.667 | 0.743/0.526 | −0.047 | −0.140 |
| p0003(同 effort) | test | 0.840/0.722 | 0.741/0.630 | −0.099 | −0.093 |
| p0003(同 effort) | ALL  | 0.874/0.746 | 0.757/0.632 | −0.117 | −0.114 |

> opt 在 **全集(ALL)** 上一致 < seed(Δavg −0.05~−0.27),不只 test 过拟合。brinks_02 效果最强、brinks_04 最弱。p0003 为唯一**同 effort**(均 xhigh)干净对比,全集 Δavg −0.117,方向/量级与四岛跨 effort 一致 → 跨 effort 参考值无系统性失真。

### 方案 A(已执行:opt full 补跑,见上;seed 仍 high)完整补跑 2214 slots

- **ev2v4_cert 全是 test-only**(train/val 均 0/30)→ 修复的 **train/val 一律从零补跑**。
- **seed persona 与 effort 无关**(指纹相同)→ seed cell 只换 effort 重跑,persona 文本不变。
- 单 cell 满编 = train30 + val30 + test54 = **114 task × 3 trial = 342 slots**。

| cell | persona(修后) | test 可复用源 | test | train | val | 补跑 |
|---|---|---|---|---|---|---|
| brinks_00_opt | `a7020ec9a7` | ev2v4_cert/b00_top1 ✓54 | 0 | 90 | 90 | 180 |
| brinks_02_opt | `c655bad793` | ev2v4_cert/b02_top1 ✓54 | 0 | 90 | 90 | 180 |
| brinks_03_opt | `47c61c2629` | ev2v4_cert/b03_top1 ✓54 | 0 | 90 | 90 | 180 |
| brinks_04_opt | `e7405d0aa9` | ev2v4_cert/b04_top1 ✓54 | 0 | 90 | 90 | 180 |
| brinks_00_seed | `653e028864` | ev2v4_cert/seed_00 ✓54 | 0 | 90 | 90 | 180 |
| brinks_02_seed | `771b2eaebd` | 无 xhigh | 162 | 90 | 90 | 342 |
| brinks_03_seed | `57c2198f5b` | 无 xhigh | 162 | 90 | 90 | 342 |
| brinks_04_seed | `82a54eb7e1` | seed_04 仅 1trial/task(seed 626729) | 108 | 90 | 90 | 288 |
| shared_no_persona | None | 无 xhigh | 162 | 90 | 90 | 342 |

**合计:opt 720 + seed 1152 + no_persona 342 = 2214 slots**
(若 brinks_04_seed 的 test 整段重跑不 salvage,则 2268;差 54)。
几乎全是全新 xhigh 生成(非 salvage),墙钟量级约 5–8h(粗估)。
**p0003(seed/opt)不动** —— 已是 xhigh 全齐(train30/val30/test54)。

> 以下「telecom high」章节的配置/winner/数据源/结论均为**旧错误版**,保留仅供审计对照,修复前勿采信。

## 机制一句话

每个 `(domain, island, condition)` cell:
1. **assemble** —— 按 **persona 指纹**(`_h` = md5(strip)[:10] of `info.user_info.persona_config.custom_guidelines`)
   + **config gate**(agent_llm / user_llm / user reasoning_effort 必须一致)吸收磁盘上已有 sim,
   只保留 canonical-seed trial(`config.seed=300` → `random` 前 3 个 = `[626729, 373753, 361454]` = trial 0/1/2),
   按 `(trial, task_id, seed)` 去重,写预填 checkpoint。
2. **run** —— `tau2 run --num-trials 3 --seed 300 --auto-resume`,`--task-ids` = train∪val∪test 全集。
   auto-resume 识别预填的 `(trial,task_id,seed)` 只补缺口。
3. 可**断点续跑**:assemble 也吸收自身输出(resume_path),崩溃重跑不丢已补 trial。

## 每域配置(config gate 的判据)

| 域 | agent_llm | user_llm | user effort | retrieval | timeout | cli domain |
|---|---|---|---|---|---|---|
| airline | gpt-5.4 | gpt-5.4 | high | — | — | airline |
| retail | gpt-5.4 | gpt-5.4 | high | — | — | retail |
| banking | **gpt-5.5** | gpt-5.4 | high | **bm25** | 600 | **banking_knowledge** |
| telecom | gpt-5.4 | gpt-5.4 | **xhigh**(🚨旧建错为 high,待修,见顶部纠错) | — | 600 | telecom |
| telecom_p0003 | gpt-5.4 | gpt-5.4 | **xhigh**(agent 仍 high) | — | 600 | telecom |

> **p0003 是独立设置**:user **xhigh** / agent **high**,单独 domain `telecom_p0003`(`split_key=telecom` 复用同一 split),汇总时自成一张表,不与 high 的 brinks 混。

> 🚨 **以下 telecom=high 论述已作废**(见顶部纠错):telecom 应为 **xhigh**,权威 test 在 `ev2v4_cert`。
> ~~telecom = high 关键:cert 目录 `ev2v4_cert`(user xhigh)只跑过 seed_00(3t)+ seed_04(1t)。
> 4 个岛完整的 3-trial seed test 都在 user high 的 `test_enriched_3seeds/`,train/val 2-trial
> salvage 在 `enriched_<isl>_g00_seed{,_train}/`(也是 high)。故 telecom 全集统一到 high。~~

## Scope(28 cells)

- **airline**:alaska_00 / alaska_01 × {seed, opt} + shared no_persona = **5 cells**
- **retail**:gc_02 / gc_03 / gc_04 × {seed, opt} + shared no_persona = **7 cells**
- **banking**:**只 cw_03 + cw_04**(cw_00/01 已排除)× {seed, opt} + shared no_persona = **5 cells**
- **telecom**(high):brinks_00/02/03/04 × {seed, opt} + shared no_persona = **9 cells**
- **telecom_p0003**(xhigh):p0003 × {seed, opt} = **2 cells**

## 每 cell 数据源 + 补跑量

split 规模:airline 10/10/30(=50),retail 15/15/84(=114),banking 20/20/57(=97),telecom 30/30/54(=114)。
每 cell 满额 = split_task_count × 3。

### airline(每 cell 满额 150)
| cell | persona_h | test 源 | test salvage | train/val salvage | 补跑 |
|---|---|---|---|---|---|
| alaska_00_seed | 3fb356be79 | `airline_topdelta_test/seed_alaska_00` | 30×3 ✓ | 无(preamble 不一致作废) | 60 |
| alaska_00_opt | c384019f56 | `airline_topdelta_test/best_alaska_00` | 30×3 ✓ | 无(opt 不可拼) | 60 |
| alaska_01_seed | a611ec744f | `airline_topdelta_test/seed_alaska_01` | 30×3 ✓ | 无 | 60 |
| alaska_01_opt | b8218f0de3 | `airline_topdelta_test/best_alaska_01` | 30×3 ✓ | 无 | 60 |
| shared_no_persona | None | `airline_topdelta_test/no_persona` | 30×3 ✓ | 无 | 60 |

### retail(每 cell 满额 342)
| cell | persona_h | test 源 | test salvage | train/val salvage | 补跑 |
|---|---|---|---|---|---|
| gc_02_seed | e46af24d37 | `retail_topdelta_test/seed_gc_02` | 84×3 ✓ | 无 | 90 |
| gc_02_opt | c23d89b103 | `retail_topdelta_test/best_gc_02` | 84×3 ✓ | 无 | 90 |
| gc_03_seed | 94ace242d9 | `retail_topdelta_test/seed_gc_03` | 84×3 ✓ | 无 | 90 |
| gc_03_opt | 39359cb56f | `retail_topdelta_test/best_gc_03` | 84×3 ✓ | 无 | 90 |
| gc_04_seed | cffdc78405 | `retail_topdelta_test/seed_gc_04` | 84×3 ✓ | 无 | 90 |
| gc_04_opt | a5af103331 | `retail_topdelta_test/best_gc_04` | 84×3 ✓ | 无 | 90 |
| shared_no_persona | None | `retail_topdelta_test/no_persona` | 84×3 ✓ | 无 | 90 |

> airline/retail 的 cert test **不带** ROLEPLAY_PREAMBLE,而 search 期 train/val **带** preamble,
> 指纹不一致 → train/val 无法 salvage,全部对齐 no-preamble 重跑。

### banking(每 cell 满额 291,retr=bm25)
| cell | persona_h | test 源 | test salvage | train/val salvage | 补跑 |
|---|---|---|---|---|---|
| cw_03_seed | 4d82d7937d | `testmatrix_cardworks/cw_03_seed` | 57×3 ✓ | `isl_cw_03_g00_seed{,_train}` 各 2t | 40 |
| cw_03_opt | a652d93515 | `testmatrix_cardworks/cw_03_opt` | 57×3 ✓ | 无(opt 不可拼) | 120 |
| cw_04_seed | 513a909429 | `testmatrix_cardworks/cw_04_seed` | 57×3 ✓ | `isl_cw_04_g00_seed{,_train}` 各 2t | 40 |
| cw_04_opt | 4ed6d24a18 | `testmatrix_cardworks/cw_04_opt` | 57×3 ✓ | 无 | 120 |
| shared_no_persona | None | `testmatrix_cardworks/no_persona` | 57×3 ✓ | 无 | 120 |

### telecom high(🚨旧错误版,已作废;正确 xhigh 版见顶部纠错。每 cell 满额 342,user high)
| cell | persona_h | test 源 | test salvage | train/val salvage | 补跑 |
|---|---|---|---|---|---|
| brinks_00_seed | 653e028864 | `test_enriched_3seeds/brinks_00_seed` | 54×3 ✓ | `enriched_brinks_00_g00_seed{,_train}` train30×2 / val20×2 | 80 |
| brinks_02_seed | 771b2eaebd | `test_enriched_3seeds/brinks_02_seed` | 54×3 ✓ | `enriched_brinks_02_g00_seed{,_train}` | 80 |
| brinks_03_seed | 57c2198f5b | `test_enriched_3seeds/brinks_03_seed` | 54×3 ✓ | `enriched_brinks_03_g00_seed{,_train}` | 80 |
| brinks_04_seed | 82a54eb7e1 | `test_enriched_3seeds/brinks_04_seed` | 54×3 ✓ | `enriched_brinks_04_g00_seed{,_train}` | 80 |
| brinks_00_opt | 322b27e381 | `test_gpt54user/valtop_brinks_00_idx3`(仅 1t) | 54×1 | `enriched_brinks_00_g*`(指纹极少命中) | 288 |
| brinks_02_opt | afceefc01b | `test_enriched_3seeds/brinks_02_idx3` | 54×3 ✓ | `enriched_brinks_02_g*` train42 / val40(指纹 gate) | 98 |
| brinks_03_opt | 469b445118 | `test_enriched_3seeds/brinks_03_idx13` | 54×3 ✓ | `enriched_brinks_03_g*` train10 / val40 | 130 |
| brinks_04_opt | 79e7796905 | `test_enriched_3seeds/brinks_04_idx12` | 54×3 ✓ | `enriched_brinks_04_g*` train10 / val40 | 130 |
| shared_no_persona | None | `nopersona_gpt54/test`(54×1t) | 54×1 | `nopersona_gpt54/val`(30×1t);无 train | 258 |

> seed 每 cell 补 80 = train 补第 3 trial(30)+ val(salvage 只覆盖 20/30 × 2t)补 50。
> opt salvage 靠 **persona 指纹 gate** 从 `enriched_<isl>_g*` 广扫里挑出 winner 的 sim(GEPA 记了指纹)。
> **brinks_00 opt** 无 3-trial cert(其 val-top 候选 test 仅 1t)→ test 重跑到 3t,故补 288。

### telecom_p0003(每 cell 满额 342,user **xhigh** / agent high)
| cell | persona_h | test 源 | test salvage | train/val salvage | 补跑 |
|---|---|---|---|---|---|
| p0003_seed | 2af9017cec | `ev2v4_cert/p0003_seed` | 54×3 ✓ | `ev2v4_p0003_g*` train60 / val40(指纹 gate) | 80 |
| p0003_opt | 5dd389d157(best) | `ev2v4_cert/p0003_best` | 54×3 ✓ | `ev2v4_p0003_g*` train42 / val40 | 98 |

> winner 判据 = val_agg 最高(见 `test_enriched_3seeds/summary.json` 的 `source`+`val_agg`);
> brinks_03 idx9/idx13 val_agg 并列 0.3,取最终代 gen=20 的 **idx13**。
> p0003 opt = cert `p0003_best`(GEPA 选中 winner,指纹 5dd389d157)。

## 补跑总量

| 域 | cells | 每 cell 补 | 小计 |
|---|---|---|---|
| airline | 5 | 60 | 300 |
| retail | 7 | 90 | 630 |
| banking | 5 | 40/120 | 440 |
| telecom seed | 4 | 80 | 320 |
| telecom opt | 4 | 288/98/130/130 | 646 |
| telecom no_persona | 1 | 258 | 258 |
| telecom_p0003 | 2 | 80/98 | 178 |
| **合计** | **28** | | **2772** |

> test 大部分靠拼(seed/opt cert + salvage,共 ~3800 slot 免跑)。
> **已于 2026-07-14 全部补跑完成**(首轮 21 cell + 第二轮 telecom 7 cell 1082 slots;含 banking task_012 的 2 条 infra_error 重补)。

## 结果汇总(avg@3 / pass^3,`fullset_summary.py` 生成)

口径:**avg@3** = 每 task 3 trial 均 reward 再对 task 平均;**pass^3** = 3 trial 全 reward==1 的 task 占比。

**test-split**

> 🚨 下表「telecom high」列作废(错误配置);正确 xhigh 版待修复后重出。

| cell | airline | retail | banking | telecom high(🚨作废) |
|---|---|---|---|---|
| seed | a00 0.856/0.767, a01 0.844/0.767 | gc02 0.841/0.690, gc03 0.841/0.702, gc04 0.861/0.738 | cw03 0.316/0.193, cw04 0.287/0.175 | b00 0.660/0.481, b02 0.827/0.704, b03 0.809/0.593, b04 0.772/0.648 |
| opt | a00 0.833/0.700, a01 0.878/0.800 | gc02 0.786/0.631, gc03 0.817/0.690, gc04 0.810/0.655 | cw03 0.216/0.105, cw04 0.240/0.105 | b00 0.735/0.574, b02 0.747/0.556, b03 0.765/0.685, b04 0.784/0.685 |
| no_persona | 0.867/0.800 | 0.817/0.655 | 0.269/0.105 | 0.815/0.704 |

**telecom_p0003(xhigh)test-split**:seed 0.840/0.722、opt 0.741/0.630。

**全集(train+val+test 合并)域内均值 Δ(opt−seed)**:airline +0.003/+0.000(打平)、retail −0.038/−0.047、banking −0.045/−0.067。→ opt 在留出集不优于 seed,与 test-set 过拟合塌缩一致。

> 🚨 **以下 telecom brinks 的 test-split 数字全部作废**(基于错误的 high 配置 + 错 persona)。
> 正确 xhigh(ev2v4_cert)下 opt diff 明显 > seed(0.278–0.438),opt 其实有效;修复后重算。p0003 数字有效。

**telecom test-split Δ(opt−seed | opt−no_persona)**(⚠️旧错误版):
- brinks_00 opt−seed **+0.074/+0.093**、opt−no_p −0.080/−0.130
- brinks_02 opt−seed **−0.080/−0.148**、opt−no_p −0.068/−0.148
- brinks_03 opt−seed −0.043/+0.093、opt−no_p −0.049/−0.019
- brinks_04 opt−seed +0.012/+0.037、opt−no_p −0.031/−0.019
- **p0003(xhigh)** opt−seed **−0.099/−0.093**(telecom 里 opt 相对 seed 最见效的一条)

→ telecom opt vs seed 混合、量级小(均值 ≈ −0.009);**opt vs no_persona 全负** —— opt 甚至不如无 persona。整体印证过拟合塌缩。

## 补全性审计(2026-07-14,全 28 cells 通过)

审计脚本对每 cell 检查:

1. **trial↔seed 配对**:每个预填 sim 的 `seed` 必须落在 CANON_SEEDS,且 `trial == CANON_SEEDS.index(seed)`
   —— 否则 auto-resume 会 skip 错 / 重复。**结果:全 0 错配。**
2. **预填 tasks ⊆ run task-ids**:否则 tau2 报 removed-task ValueError。**结果:全 True。**
3. **split 全覆盖**:每个 train/val/test task 都在 `--task-ids`(=三 split 并集)里,
   保证任何 <3 trial 的 task 都会被补到 3。**结果:全 True。**

**结论:所有 cell 均可完整补全到每 task 3 trial。** smoke(banking cw_03_seed)已实证:
预填 7 trial 全被跳过、只补 2 个缺口,最终 3 个 task 均 [trial0,1,2] 齐全。

## ✅ 数据完整性(2026-07-14,已全部补齐)

相对"每域 seed + opt + no_persona 全 full(train+val+test @3t)"这个完整矩阵:

- **airline / retail / banking** —— 全 full(banking cw_00/cw_01 是主动砍掉,不算缺)。
- **telecom brinks** —— 🚨 **建错待修**(9 cell 用了 high+错 persona,应 xhigh+ev2v4_cert winner;修复补跑 2214 slots,见顶部纠错)。旧「high 全 full」不作数。
- **telecom_p0003(xhigh)** —— seed + opt 全 full(**同轮补完**)。

**唯一未覆盖**(有意为之,非缺口):no_persona 只做了 high 那份;p0003 无 no_persona(p0003 是单岛 seed-vs-opt 对比,不需要)。

### winner 选择口径(telecom opt)

winner = **val_agg 最高**(见 `test_enriched_3seeds/summary.json` 的 `source`+`val_agg`):
brinks_02→`idx3`(0.325)、brinks_03→`idx13`(0.3,与 idx9 并列取 gen=20)、brinks_04→`idx12`(0.375)。
**brinks_00 无 3-trial cert**(`test_enriched_3seeds` 里只有 seed),取其 val-top 候选 `test_gpt54user/valtop_brinks_00_idx3`(指纹 322b27e381,仅 1t)→ test 重跑到 3t。
p0003 opt = cert `ev2v4_cert/p0003_best`(GEPA 选中 winner,指纹 5dd389d157)。

> ⚠️ 早前 doc 误记 brinks_00→`idx6` —— `idx6` 实为 **brinks_01** 的候选(brinks_01 不在 scope)。已更正。

### telecom opt/seed 的 salvage 机制(与其它域不同)

telecom 的 GEPA run **序列化了 persona 指纹**,所以 opt/seed 的 train/val 都靠 assemble 的 **persona-hash gate** 从广扫 glob 里挑出对应 winner 的 sim:
- seed:`enriched_<isl>_g00_seed{,_train}` / `ev2v4_p0003_g00_seed{,_train}`
- opt:`enriched_<isl>_g*/**` / `ev2v4_p0003_g*/**`(gate 只留 winner 指纹的那几条)

brinks_02 opt 命中 82/180、p0003_opt 命中 82/180 等,显著少跑;brinks_00 opt 指纹几乎不命中 → 近全重跑(288)。

## 不可 salvage 的原因(为什么要重跑)

- **airline/retail/banking 的 opt train/val**:GEPA `gNN_kMM` eval 目录存的是**别的 candidate**(非 winner),
  且这些域的 dump **未记 persona 指纹**(`persona=None`)→ 无法按指纹归属,必须用 winner persona 重跑。
- **airline/retail 的 train/val**:cert test 无 preamble、search 有 preamble,指纹不一致 → 重跑对齐 no-preamble。
- **telecom fullset_brinks dump**:persona_config=None 且 user 是 claude/bedrock(非 gpt-5.4)→ 被 gate 剔除。
- **早期 no_persona 混源**:`full_gpt54_baseline`/`no_persona_*full114` 是 cc_user/cc_user_con(sonnet/opus)、`hard_no_persona`/`test_eval` 是 gpt-5.2 → 与 fullset 的 gpt-5.4 不同源,不可用;只 `gepa_opt/nopersona_gpt54/{test,val}` 可用(high,各 1t)。

## 运行 / 续跑

```bash
# 全量(顺序串行,自动 resume)
bash scripts/persona_opt/run_fullset.sh          # 日志 /tmp/fullset_run.log

# 仅报告缺口不跑
.venv/bin/python scripts/persona_opt/fullset_build.py --dry-run

# 单 cell / smoke
.venv/bin/python scripts/persona_opt/fullset_build.py --cells telecom:brinks_02:seed
.venv/bin/python scripts/persona_opt/fullset_build.py --cells banking:cw_03:seed --smoke 1

# 只写预填 checkpoint 不跑 tau2
.venv/bin/python scripts/persona_opt/fullset_build.py --assemble-only
```

崩溃后直接重跑 `run_fullset.sh` 即可:已完成 cell 的输出会被 resume_path 秒吸收,不重跑。
