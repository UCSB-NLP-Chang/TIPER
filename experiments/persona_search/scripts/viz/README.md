# Persona-Opt Visualizations

可视化 GEPA persona 优化过程的图表代码。**每个图一个独立脚本、独立输出 PNG**，方便单独改 style。

## 运行

```bash
# 全部图，两条 run
.venv/bin/python scripts/persona_opt/viz/run_all.py

# 指定 run / 输出目录 / 只跑某几张
.venv/bin/python scripts/persona_opt/viz/run_all.py --runs p0001 --out-dir /tmp/figs
.venv/bin/python scripts/persona_opt/viz/run_all.py --only plot_pareto plot_termination

# 单独跑一张（也支持 --runs / --out-dir）
cd scripts/persona_opt/viz && ../../../.venv/bin/python plot_difficulty_trajectory.py
```

输出：`data/persona_opt/figs/<run>/<run>_<plot>.png`

## 结构

| 文件 | 作用 |
|------|------|
| `_style.py` | **全局风格**：palette、rcParams、DPI、figsize。改一处变全部图 |
| `_data.py` | 数据加载：gepa_state.bin（候选级）+ eval results.json（sim/message 级）。自动处理 unpickle 的 sys.path |
| `plot_*.py` | 每图一个脚本，含 `build(run_id, out_dir)` + 顶部局部 STYLE 块（只改这张图） |
| `run_all.py` | 批量跑全部图 |

## 图清单

| 脚本 | 看什么 | 数据源 |
|------|--------|--------|
| `plot_difficulty_trajectory` | difficulty 随 budget 单调爬升 + 每候选 validity 配色 | gepa_state |
| `plot_pareto` | difficulty vs validity 散点 + Pareto front | gepa_state |
| `plot_objective_breakdown` | 每候选的 difficulty/validity/timeout_safety/semantic/feasible 分项 | gepa_state |
| `plot_lineage_tree` | 候选谱系（parent→child 突变树），节点色=difficulty | gepa_state |
| `plot_pertask_heatmap` | 候选×50任务的 per-task 难度热图 | gepa_state |
| `plot_mutation_acceptance` | 每 iteration parent vs child minibatch 难度 + 接受/拒绝 | gepa_state |
| `plot_tokens_per_turn` | avg completion tokens/turn（user vs agent）+ 随轮次趋势 | eval sims |
| `plot_message_length` | 消息字符长度分布（user/agent/tool） | eval sims |
| `plot_conversation_length` | 对话长度分布，按 agent 成功/失败拆分 | eval sims |
| `plot_termination` | termination reason 随轮次堆叠（兼数据质量监控：timeout 飙升=SSO挂） | eval sims |

## 改 style

- 全局（配色/字号/DPI/网格）：改 `_style.py` 的 `PALETTE` 和 `apply_style()`。
- 单张图（某个图特有的尺寸/标注/bin 数）：改该 `plot_*.py` 顶部的局部常量与 `build()`。

## 注意

- 数据随优化推进变化；优化跑到 700/停后重跑会反映最终状态。
- gepa_state.bin 只在 iteration 边界更新；eval results.json 每轮追加。
- 字段细节见 `_data.py` 顶部注释（reward 在 `reward_info.reward`，tokens 在 `usage.completion_tokens` 等）。
