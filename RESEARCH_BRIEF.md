# Research Brief: 复读抑制机制的跨规模演变

## Research Direction

在 Qwen3 系列 ≤10B 模型中追踪复读抑制机制随规模的演变，验证 "小模型 focal single-head RSH → 大模型 distributed suppression" 的 mechanistic scaling hypothesis。

## Background: 分层事实 (Phase 1–6)

### Strong evidence（多实验交叉验证）

1. 在 1.7B 的 greedy decoding + zero-ablation 设置下，全量 448-head 消融中**仅 L0H3** 被屏蔽后稳定引发复读（rep rate 0%→99%）。这是特定实验条件下的结论，不等价于"L0H3 是唯一 RSH"。
2. L0H3 **不符合经典 copy suppression 定义**（静态 OV top-10 suppression=0%）。但这只排除了明显的静态 top-rank OV 抑制，不排除 context-conditional 或 LN-amplified 的间接 logit 效应。
3. L0H3 98.6% self-attention：output ≈ position-dependent bias vector。
4. LoopBench v3 baseline：1.7B=22.71%, 4B=21.43%。
5. LoopBench 严格单调筛选不适合做主发现器（floor effect）。

### Moderate evidence（需进一步验证）

6. 0.6B 存在**功能类似的 analogous head**（同为 L0H3，rep delta +0.93）。尚无 weight-space alignment 验证。
7. 4B **在当前筛选方法下**未找到 clean single-head RSH。这可能反映 (a) 发现方法的盲区，(b) 抑制功能分布式化，或 (c) RSH 不在 Layer 0。

### Working hypothesis（待因果验证）

8. L0H3 的作用机制**可能**是间接的：通过改变 token 竞争格局影响复读（描述性标签，需 causal mediation 确认）。

## Paper 主叙事（Circuit Identification + Mechanistic Scaling）

**线索 A — Circuit Identification（类比 Induction Heads 方法论）**

1. 定义 per-head 评分函数（DLA, SAC, DCS），在不做 ablation 的情况下定位 RSH 候选 + 标识 circuit 成员
2. Ablation 确认因果必要性，与评分函数交叉验证
3. Per-component logit attribution + path patching 追踪 circuit 路径（目标：产出可视化 circuit diagram）
4. 明确与 copy suppression 的区别

**线索 B — Mechanistic Scaling**

1. 在两个小模型（0.6B/1.7B）中都观察到类似的 Layer-0 suppressor-like head（功能类比，非 weight-space correspondence）
2. 到 4B，当前方法下的单头可定位性下降 → DLA landscape 集中度变化揭示抑制功能分布模式
3. 即使没有 clean single-head RSH，DLA landscape 和 circuit 组件的跨模型比较仍能支撑 scaling 叙事

## Core Research Questions

1. **Q0 — Circuit identification**: L0H3 参与的具体 circuit 是什么？（self-attend → output bias → 哪些下游组件？）
2. **Q1 — Scaling transition**: 复读抑制机制在 focal → distributed 光谱上如何随规模演变？DLA landscape 的集中度如何变化？
3. **Q2 — 因果路径**: L0H3 的具体中介路径是什么？是 norm 维持、方向偏移、还是 MLP 门控？
4. **Q3 — 特异性与稳健性**: RSH 效果是否特异于病态复读？是否依赖特定解码配置？

---

## 方法论核心（借鉴 Induction Heads, Olsson et al. 2022）

三条独立证据流，类比 Induction Heads 的方法论：

| 证据流 | Induction Heads 方法 | 本项目 RSH 方法 | 实验 |
|-------|---------------------|----------------|------|
| **Mechanistic scoring** | prefix-matching + copying score | DLA + SAC + DCS 评分函数 | Exp 0 |
| **Causal intervention** | per-head ablation + ICL loss | zero-ablation + rep rate | Exp 1, 2 |
| **Circuit tracing** | K-composition, 2-step circuit | per-component logit attribution + path patching + DCS | Exp 3, 4 |
| *Cross-condition* | training dynamics phase change | cross-model scaling comparison | Exp 5 |

## Operational Definitions（全局统一）

| 术语 | 定义 | 适用实验 |
|------|------|---------|
| `tok_repeat(t)` | position t-1 实际生成的 token（在 baseline trajectory 上） | Exp 0, 3, 4 |
| `tok_best_alt(t)` | position t 的 baseline logit 分布中，排除 `tok_repeat(t)` 后 logit 最高的 token；tie-breaking: token id 最小者 | Exp 0, 3, 4 |
| `logit_margin(t)` | `logit(tok_repeat(t)) - logit(tok_best_alt(t))`，正值表示模型倾向复读 | Exp 0, 3b-0, 3b-4 |
| `collapse boundary` | baseline trajectory 正常但 zero(L0H3) trajectory 首次进入 loop 的 position ± 5 tokens | Exp 3b-4, Exp 0 Mode B |
| `repeat-prone position` | baseline 下 `tok_repeat(t)` 在 logit top-3 内的 position（不依赖 ablation） | Exp 0 Mode A |
| `recovery rate` | `1 - Δrep_patched / Δrep_zero`；edge case: 如果 `Δrep_zero = 0`，该样本排除 | Exp 3b-1, 3b-2 |
| `mediation %` | `(logit_margin_zero - logit_margin_patch) / (logit_margin_zero - logit_margin_baseline) × 100%`；100% = 该路径完全恢复 baseline，0% = 该路径无效；如果分母 \|total_effect\| < 0.1，排除 | Exp 3b-4 |
| `mixed circuit` | H-attn 和 H-mlp 两条路径的 mediation 均在 30-60% 且差异不显著 (paired bootstrap p > 0.01) | Exp 3b-4 判别 |
| `mixed contribution` | norm 和 direction 的 recovery rate 均在 30-60% 且差异不显著 (p > 0.01) | Exp 3b-1/3b-2 判别 |

## 结构化执行表 (ARIS Machine-Actionable)

### 全局统计规范

| 参数 | 值 |
|------|-----|
| Bootstrap 次数 | B = 10000 |
| Bootstrap 方法 | prompt-level resampling（按 prompt 重采样，每次抽 N 个 prompt with replacement） |
| 假设检验方向 | 单侧（H0: mean Δrep ≤ 0, H1: mean Δrep > 0） |
| FDR 方法 | Benjamini-Hochberg |
| FDR alpha | 0.05 |
| Family 定义 | 每个模型 × 每个 stage 的全部 head |
| 效应量 | Cohen's d（paired） |
| 输出 | 每个 head: mean Δrep, 95% CI, raw p, BH q-value, Cohen's d |

### Exp 0: 全头机制评分 (Dense Mechanistic Landscape)

| Step | Trigger | Inputs | Operation | Statistic | Pass/Fail | Next |
|------|---------|--------|-----------|-----------|-----------|------|
| **0.0 校准** | always | 1.7B, LoopBench v3 square_root 50 题 | 全 448 heads 计算 DLA (Mode A, ablation-independent) + SAC | L0H3 DLA rank, recall@k (k=1,3,5,10) | **Pass**: L0H3 DLA rank ≤ 10 | 0.1 |
| | | | | | **Informative fail**: DLA rank > 10 → 间接机制支持，DLA 降格为 landscape 工具 | Exp 5a only |
| **0.1 DCS** | always (纯 weight 计算，不依赖 DLA 结果) | 1.7B L0H3, Layer 1-5 全部 heads | 计算 DCS_K 和 DCS_V | top downstream heads by DCS | 标记 circuit candidate edges | Exp 3b-4 交叉验证 |
| **0.2 扩展** | always (DLA landscape 用于 Exp 5a 跨模型比较，不论 0.0 校准结果) | 4B/7B/8B 全 heads | 同 0.0 protocol (Mode A) | DLA landscape per model | 输出 landscape 热力图; 若 DLA rank ≤ 10 的 head 不在 L0 → 纳入 Stage 1 覆盖范围 | Exp 5a + Exp 1 层补充 |

### Exp 1: 跨模型 RSH 定位

| Step | Trigger | Inputs | Operation | Statistic | Pass/Fail | Next |
|------|---------|--------|-----------|-----------|-----------|------|
| **1.0 校准** | always | 1.7B, LoopBench v3 square_root 50 题 | Stage 0 + Stage 1 full protocol on 1.7B | L0H3 的 BH q-value 和 rank | **Pass**: BH q < 0.05 AND rank ≤ 3 (by q ascending) | 1.0-CAL → 1.1 |
| | | | | | **Fail**: L0H3 missed | 调整 protocol（增 sample size 或降 Stage 0 粒度），重跑 |
| **1.0-CAL** | 1.0 pass | 1.7B Stage 1 全部 head 的 q-values + Exp 0 DLA ranks + gold label (L0H3=pos) | recall@k 对比表 (DLA vs q-value vs prev-1 attn vs wait logit), k={1,3,5,10} | 各方法 recall@k + L0H3 rank percentile | 输出报告 | 大模型筛选：BH q < 0.05 (主) + DLA top-10 (辅) |
| **1.1 Stage 0** | 1.0 pass | 4B/7B/8B, LoopBench v3 square_root 50 题 | 每层 3 轮 × 4 head 随机哨兵 (min(12, H_per_layer) heads/layer) | layer-level 热力图 | 标记有信号的 layer | 1.2 |
| **1.2 Stage 1** | 1.1 done | 目标层（L0 + 信号层）全部 head, 100 题 | 全 head zero-ablation + 统计流程 | BH q-value per head | **Pass**: ≥1 head BH q < 0.05 | 1.3 |
| | | | | | **Fail**: 无 head BH q < 0.05 | Exp 2 |
| **1.3 Stage 2** | 1.2 pass | 候选 head, LoopBench full 700 + NuminaMath 100 | scale sweep {0, 0.5, 1.0, 1.5, 2.0} | Δrep + Δacc | **Pass**: ablation ↑rep (p<0.05) AND scale>1 不 ↓acc | Exp 5 (if new model) |
| | | | | | **Fail**: 不复现 | Exp 2 (trigger 3: 噪声假阳性) |

### Exp 2: 分布式假说测试

| Step | Trigger | Inputs | Operation | Statistic | Pass/Fail | Next |
|------|---------|--------|-----------|-----------|-----------|------|
| **2a** | Exp 1 Stage 0 有 deeper-layer 信号 | 信号层全部 head, 100 题 | 同 Stage 1 protocol | BH q-value | 有 BH q < 0.05 → Exp 1.3 | — |
| **2b** | Exp 1 无信号 OR 多头弱信号 OR 噪声假阳性 | top-k heads (k={2,3,5,8}), discovery 100 题 + **held-out 50 题** | grouped ablation + 3 种 matched controls × 20 perm + additivity test | group Δrep vs controls (Bonferroni p<0.01); synergy = group Δrep / sum(single Δrep) | **Pass**: held-out 复现 + group > all 3 controls | 2c |
| | | | | | **Fail**: held-out 不复现 or group ≤ controls | 结论：attention head 非主要承载层 → MLP/SAE |
| **2c** | 2b pass | top-k group, held-out 50 题 | leave-one-out | 边际贡献 per head | head 重要性排序 | Exp 5 |

### Exp 3: L0H3 Causal Mediation

| Step | Trigger | Inputs | Operation | Statistic | Pass/Fail | Next |
|------|---------|--------|-----------|-----------|-----------|------|
| **3A** | always | 1.7B L0H3, 100 样本 | 全样本描述：norm, cosine, entropy, LN scale (baseline vs zero) | 分层对比 (loop vs non-loop) | 描述性报告 | 3B |
| **3B-0** | 3A done | 100 样本, collapse-boundary positions | per-component logit attribution: 每个 head + MLP layer 对 repeat-vs-alt logit diff 的贡献 (baseline vs zero(L0H3)) | 全景 attribution 图; Δattribution per component | 标识 circuit 成员: ablation 后 attribution 变化最大的组件 | 3B-4 交叉验证 |
| **3B-1** | 3A done | 100 样本, post-attn pre-MLP residual | norm-preserving direction patch (per-token) | recovery rate = 1 - Δrep_patched/Δrep_zero | report rate | 比较 3B-1 vs 3B-2 |
| **3B-2** | 3A done | 同上 | direction-preserving norm patch (per-token) | recovery rate | report rate | 比较 |
| **3B-3** | 3A done | 同上 | project-out L0H3 mean direction | Δrep_projectout vs Δrep_zero | 效应集中度 | — |
| **3B-4** | 3A done | collapse boundary ±5 tokens | path patching: Path A (H-attn: L0H3→downstream attn) vs Path B (H-mlp: L0H3→MLP0) | mediation % of repeat-vs-alt logit diff | H-attn: mediation_A > 60%; H-mlp: mediation_B > 60%; mixed circuit: both 30-60%; 与 Exp 0 DCS 交叉验证 | 3B-5 |
| **3B-5** | 3B-0 + 3B-4 done | 3B-0 attribution + 3B-4 patching + Exp 0 DCS | circuit diagram synthesis | 组件贡献之和 ≥ 80% of total repeat logit diff 变化 | **Pass**: 可画 circuit diagram | Exp 6 |
| | | | | | **Fail**: 贡献散布，无法汇聚 | 报告为 diffuse mechanism |
| **3B 判别** | 3B-1 + 3B-2 done | recovery rates | paired bootstrap (B=10000) of rate difference | p-value of diff | **Distinct** (p<0.01): norm vs direction 主导因子确定 | Exp 6 |
| | | | | | **Mixed contribution** (both 30-60%, diff ns p>0.01) | Exp 6 |
| **Null** | 3B done | 10 random L0 heads × same interventions | permutation test | p < 0.01 | 必须 pass | — |

### Exp 4, 5, 6: 简表

| Exp | Trigger | Core Operation | Key Metric | Pass Criterion |
|-----|---------|---------------|------------|----------------|
| **4** | always | L0H3 output → unembed → repeat-vs-alt logit diff | PCA variance, pairwise cosine, logit diff | fixed bias vs context-dependent 判别 |
| **5** | Exp 1 finds RSH in new model | Repeat Exp 3/4 core on new RSH | ≥2 shared mechanism features | 保守 vs 独立机制 |
| **6a** | Exp 3B done | zero-ablation on "正常重复" 50 samples | Δ generation quality | RSH ablation 无显著影响 → 特异性 |
| **6b** | Exp 3B done | zero-ablation × 3 decoding configs (greedy/sample/nucleus) | Δrep consistency | ≥2 configs 方向一致且 CI 不跨零 |

---

## Key Confounds to Control

| Confound | Risk | Mitigation | Experiment |
|----------|------|-----------|------------|
| Zero-ablation off-manifold | 中高 | Exp 3B-3 project-out; Exp 6b 多配置 | 3, 6 |
| 生成长度 ↔ loop 检测耦合 | 中 | 区分 "恢复正常" vs "打满 max_tokens" | all |
| High-loop subset → task-specific | 中 | Stage 2 full 700 题; Exp 6a 多题型 | 1, 6 |
| 跨模型 head 数不同 | 低中 | per-prompt paired design | 1 |
| 解码配置伪影 | 中 | Exp 6b (greedy/temp=0.6/temp=1.0) | 6 |
| Stage 0 false negative | 中 | 检出力声明；负结果不作排除证据 | 1 |
| Exp 2 top-k winner's curse | 中 | held-out prompt set 验证 | 2 |

## 替代假说（尚未排除）

| 假说 | 测试实验 | 排除标准 |
|------|---------|---------|
| 分布式多头冗余 | Exp 2b (grouped + additivity) | group > controls on held-out + synergy/additive |
| H-mlp (MLP pathway 为主) | Exp 3B-4 (path patching) | mediation_B (Path B) > 60% → 采纳 H-mlp |
| Attention sink / value drain | Exp 3A (与 pure-sink heads 对比) | L0H3 效应显著异于 pure-sink |
| LayerNorm 门控 | Exp 3A (LN scale 追踪) | LN scale Δ 与 loop outcome 相关 |
| 任务/题型特异 | Exp 6a | ablation 效应跨题型一致 |
| 解码伪影 | Exp 6b | 效应跨配置一致 |

## Model & Infrastructure

- **Models**: Qwen3-0.6B, 1.7B, 4B, 7B, 8B
- **Remote server**: `ssh -p 8002 yucheng@101.6.96.183`, GPUs 0-8
- **Code**: `cot_research/` framework + `scripts/` experiment scripts
- **Data**: `evaluation/data/loopbench_reconstructed_v2/test.jsonl` (700 题)

## Evaluation Criteria

- **Exp 0 校准**: L0H3 DLA (Mode A) rank ≤ 10；如 rank > 10 则 DLA 降格为 landscape 工具
- **1.7B 校准**：L0H3 的 BH q < 0.05 且 rank ≤ 3；recall@k 对比报告完成
- **Q0 (Circuit)**: Exp 3B-5 的 circuit diagram 覆盖 ≥80% 的 repeat logit diff 变化；Exp 0 DCS 与 3B-4 path patching 结果一致
- **Q1 (Scaling)**: ≥1 个 ≥4B 模型找到 BH q < 0.05 的单头 RSH，或 Exp 5a DLA 集中度显示系统性跨规模变化
- **Q2 (Mechanism)**: Exp 3B-1/3B-2 recovery rate 判别成功 (p<0.01) 或确认 mixed contribution；Exp 3B-4 path mediation 判别成功或确认 mixed circuit
- **Q3 (Robustness)**: Exp 6b Δrep 在 ≥2 种解码配置下一致

## Constraints

- GPU 预算：~55h（含校准、条件分支、held-out 验证）
- 统计：全部使用统一规范（见全局统计规范表）
- Stage 0 负结果不可作为排除证据

## Target Venue

ICLR 2027 或 ACL 2026 ARR December
