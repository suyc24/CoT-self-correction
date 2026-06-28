# 实验索引

所有实验位于 `experiments/` 下，按研究阶段分组。  
命名规范：`{研究主题}_{模型}_{日期}_{版本号}`  
每个实验目录包含：`EXPERIMENT_TRACKER.md` / `EXPERIMENT_NOTES.md`（文档）+ `data/`（结果数据）

---

## Phase 1 — Wait Head 早期发现（2026-03-25 ~ 04-01）

> 确认 L0H3 是 1.7B 模型中最主要的 wait-suppression head；初步量化 scale 对 reflection/长度的影响。

| 实验目录 | 日期 | 模型 | 主要结论 |
|---------|------|------|---------|
| `phase1_wait_head_discovery/attention_sink_qwen3_1p7b_20260325_1` | 03-25 | 1.7B | 重复生成中 attention sink 头的早期模式分析 |
| `phase1_wait_head_discovery/head_boost_effects_qwen3_1p7b_20260325_1` | 03-25 | 1.7B | 多 sink head 在 scale=1.2 下对 NuminaMath 1K 的 boost 效果 |
| `phase1_wait_head_discovery/scale_wait_length_qwen3_1p7b_20260328_1` | 03-28 | 1.7B L0H3 | scale 0.5→1.5 将 token 数从 1826 降至 1552 |
| `phase1_wait_head_discovery/wait_head_ablation_qwen3_1p7b_20260330_1` | 03-30 | 1.7B | **L0H3 确认为首要 wait-suppression head**，ablation delta=-3.46 |
| `phase1_wait_head_discovery/wait_head_baseline_qwen3_1p7b_20260330_1` | 03-30 | 1.7B L0H3 | 8 个 baseline 样本上 suppression rate=100% |
| `phase1_wait_head_discovery/scale_reflection_lexicon_qwen3_1p7b_20260330_1` | 03-30 | 1.7B L0H3 | Lexicon 500 题：reflection 从 10.1 单调降至 7.1 |
| `phase1_wait_head_discovery/scale_reflection_numinamath_qwen3_1p7b_20260331_1` | 03-31 | 1.7B L0H3 | NuminaMath 1K：保持 ~44% 准确率，token -891 |
| `phase1_wait_head_discovery/wait_prefix_ablation_qwen3_1p7b_20260401_1` | 04-01 | 1.7B L0H3 | prefix 效果分析；第一个 wait token delta=-3.46 后递减 |

---

## Phase 2 — L0H22 在 4B 上的 Scale 效果（2026-04-05 ~ 04-06）

> 测试 4B 的类 L0H3 头（L0H22）能否复现 1.7B 的 scale 效果；结论：效果弱得多。

| 实验目录 | 日期 | 模型 | 主要结论 |
|---------|------|------|---------|
| `phase2_l0h22_scale_4b/scale_reflection_numinamath_qwen3_4b_20260405_1` | 04-05~06 | 4B L0H22 | 4B L0H22 scale 效果弱于 1.7B L0H3；准确率 39-42% 无显著变化（含 4 个 run 变体） |

---

## Phase 3 — L0H3 机制深度分析（2026-04-04 ~ 04-09）

> 系统分析 L0H3 的注意力模式、OV circuit、copy suppression 机制，以及对复读的因果影响。

| 实验目录 | 日期 | 模型 | 主要结论 |
|---------|------|------|---------|
| `phase3_l0h3_mechanism/attention_pattern_qwen3_1p7b_20260404_1` | 04-04 | 1.7B L0H3 | **L0H3 极度局部**：98.6% self-attention，prev-token mass 极小 |
| `phase3_l0h3_mechanism/prev1_attention_probe_qwen3_1p7b_20260407_1` | 04-07 | 1.7B | prev-1 注意力质量在所有 head 中的排名基准（含 merged/ 合并结果） |
| `phase3_l0h3_mechanism/head_discovery_multi_20260407_1` | 04-07 | 0.6B/1.7B/4B | **跨模型发现**：1.7B/0.6B L0H3 排名第一；4B 头为 L0H2，机制不同 |
| `phase3_l0h3_mechanism/collapse_prefix_qwen3_1p7b_20260408_1` | 04-08 | 1.7B L0H3 | scale=1.5 成功解救 108 个 loop-collapse case |
| `phase3_l0h3_mechanism/copy_suppression_qwen3_1p7b_20260408_1` | 04-08 | 1.7B L0H3 | **否定假设**：L0H3 对 copy rate 影响弱，不是直接 copy suppressor |
| `phase3_l0h3_mechanism/ov_circuit_qwen3_1p7b_20260408_1` | 04-08 | 1.7B L0H3 | OV circuit 有 50.9% 负 self-logit，但实际 top-10 suppression=0% |
| `phase3_l0h3_mechanism/copy_temptation_qwen3_1p7b_20260408_1` | 04-08 | 1.7B L0H3 | 精确边界 local copy 下 scale 效果弱 |
| `phase3_l0h3_mechanism/copy_temptation_sharp_qwen3_1p7b_20260408_2` | 04-08 | 1.7B L0H3 | 尖锐短语边界下 target-token 有改善，288 例 |
| `phase3_l0h3_mechanism/copy_temptation_semantic_qwen3_1p7b_20260408_3` | 04-08 | 1.7B L0H3 | scale 区分语义 target vs 空白 target，288 例 |
| `phase3_l0h3_mechanism/phrase_copy_qwen3_1p7b_20260408_1` | 04-08 | 1.7B L0H3 | 短语级 copy suppression 不稳定，1152 条件 |
| `phase3_l0h3_mechanism/repetition_suppression_qwen3_1p7b_20260408_1` | 04-08 | 1.7B L0H3 | 1523 个复读 case：改善 9.4%，但回归 8.1%，net 效果有限 |
| `phase3_l0h3_mechanism/repetition_causal_qwen3_1p7b_20260409_1` | 04-09 | 1.7B L0H3 | **机制结论**：L0H3 通过改变 token 竞争格局间接影响复读，非直接抑制 |

---

## Phase 4 — Scale 对 Benchmark 的影响（2026-04-09 ~ 04-13）

> 在多个 benchmark 上量化 L0H3/L0H22 scale 对正确率的实际影响；探索 4B 的候选 head。

| 实验目录 | 日期 | 模型 | 主要结论 |
|---------|------|------|---------|
| `phase4_scale_benchmark/scale_benchmark_minerva_qwen3_1p7b_20260409_1` | 04-09 | 1.7B L0H3 | Minerva 272 题：scale 1.4 轻微效率提升，准确率持平 |
| `phase4_scale_benchmark/scale_aime25_qwen3_1p7b_20260411_1` | 04-11 | 1.7B L0H3 | AIME 2025：大倍率 scale=32 下行为分析 |
| `phase4_scale_benchmark/current_top10_scale_gsm8k_qwen3_4b_20260413_1` | 04-13 | 4B | **4B current-token top10 head 扫描**：L0H9 最佳（length -190，scale 2.0） |
| `phase4_scale_benchmark/prev1_all20_scale_gsm8k_qwen3_4b_20260413_1` | 04-13 | 4B | 4B prev-1 top20 head 队列测试 GSM8K（结果待完整分析） |
| `phase4_scale_benchmark/prev1_top10_scale_gsm8k_qwen3_4b_20260413_1` | 04-13 | 4B | 4B prev-1 top10 head 测试 GSM8K 100 题 |

---

## Phase 5 — Head 局部性分类（2026-04-14 ~ 进行中）

> 系统分类 1.7B 和 4B 中所有 head 的局部注意力特征；寻找可复用的 local head。

| 实验目录 | 日期 | 模型 | 主要结论 |
|---------|------|------|---------|
| `phase5_head_locality/boundary_probe_qwen3_1p7b_20260414_1` | 04-14 | 1.7B L0H3 | L0H3 boundary probe 分类基准 |
| `phase5_head_locality/head_locality_classification_qwen3_1p7b_20260414_1` | 04-14 | 1.7B | **1.7B 全量 head 分类**：448 heads 中 29 local（L0H3 subtype=self_local） |
| `phase5_head_locality/boundary_probe_qwen3_4b_20260414_1` | 04-14 | 4B | 4B top head（L0H1, L22H12）的 boundary probe |
| `phase5_head_locality/repetition_scale_logit_qwen3_1p7b_20260415_1` | 04-15 | 1.7B | repetition 样本上的 scale-logit 关系扫描 |
| `phase5_head_locality/local_heads_scale_accuracy_qwen3_4b_20260416_1` | 04-16~17 | 4B | **除掉 9 个已经检查过的 head，34 个 local head 扫描完成**：严格候选为 `L15H10`、`L2H25`、`L1H29`、`L0H18`，其中 `L15H10` 最强 |
| `phase5_head_locality/local_heads_repetition_screen_qwen3_1p7b_20260419_1` | 04-19~20 | 1.7B | **29 个 local head 的 LoopBench 复读筛选**：严格候选仅 `L0H7/L0H5/L0H12`，但该方法因 baseline 仅 `1/100` loop 且漏掉 `L0H3`，**不适合作为主发现器** |

---

## Phase 6 — LoopBench 重建与 Baseline 校验（2026-04-21 ~ 进行中）

> 严格参考 arXiv:2601.05693 重建 LoopBench，并先验证 baseline loop rate 是否接近论文量级。

| 实验目录 | 日期 | 模型 | 主要结论 |
|---------|------|------|---------|
| `phase6_loopbench_reconstruction/loopbench_baseline_v3_qwen3_multi_20260422_1` | 04-21~22 | 1.7B / 4B | **v3 baseline 修复后复读率恢复到可信量级**：`1.7B=22.71%`、`4B=21.43%`；修复点包括 `vllm chat` 路径、`max_new_tokens=16384`、`repetition_penalty=1.0`。两个模型都出现显著 numerical loop，但 `statement loop` 仍偏少，且 `hit_max_rate≈73%~75%`，说明 benchmark 已可用于 head 筛查，但 statement-side fidelity 仍未完全解决 |
考虑放弃这条路径

---

## Phase 7 — Stateful Tampering 与 Reflection Gate（2026-05-10 ~ 进行中）

> 从 Stateful Forced Tampering 出发，先确认反思主要由 visible transcript inconsistency 触发，再构造 `tamper - coherent` residual direction，验证其能跨 Qwen3-4B / Qwen3-1.7B 控制 self-check / deliberation intensity。

阶段性报告：

| 报告 | 日期 | 主要内容 |
|------|------|---------|
| `reports/stateful_forced_tampering_qwen3_20260511.md` | 05-11 | stateful mismatch 非必要、local coherent wrong transcript 关闭 `Wait`、L19-L20 boundary patch 与 attention/source-token 证据 |
| `reports/reflection_gate_experiment_plan_20260512.md` | 05-12 | reflection gate 下一阶段实验设计 |
| `reports/reflection_gate_current_report_20260513.md` | 05-13 | 当前完整结果：4B/1.7B gate、gate-orth-logit、random/logit controls、8192 validation 状态与剩余时间预估 |
| `reports/reflection_gate_advisor_report_20260517.md` | 05-17 | 导师汇报版：重组故事线，只保留 visible local inconsistency、L19-L22 residual boundary、reflection gate 跨模型干预复现等主线证据 |

---

## 目录总览

```
experiment_results/
├── INDEX.md                         ← 本文件
├── experiments/
│   ├── phase1_wait_head_discovery/  (8 个实验)
│   ├── phase2_l0h22_scale_4b/      (1 个实验，4 个 run 变体)
│   ├── phase3_l0h3_mechanism/      (12 个实验)
│   ├── phase4_scale_benchmark/     (5 个实验)
│   └── phase5_head_locality/       (6 个实验)
│   └── phase6_loopbench_reconstruction/ (1 个实验)
├── local_current/                   ← 本地运行的少量结果
└── reports/                         ← 阶段性报告
```
