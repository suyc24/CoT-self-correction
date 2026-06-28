# Experiment Plan: 跨模型 RSH 定位与机制解释

**Date**: 2026-04-29（基于 Phase 1–6 结果全面重写）  
**Models**: Qwen3-0.6B / 1.7B / 4B / 7B / 8B  
**Core question**: 在 ≤10B 的 Qwen3 模型中系统定位 RSH，并解释其作用机制

---

## 分层事实（来自 Phase 1–6）

*Strong evidence：*

1. **1.7B L0H3** 是当前实验条件下（greedy decoding, zero-ablation, 特定 prompt set）观察到的唯一单头因果必要因子（全量 448-head 消融确认）
2. **L0H3 不符合经典 copy suppression 定义**：静态 OV top-10 suppression=0%（不排除 context-conditional 间接效应）
3. **L0H3 98.6% self-attention**：output ≈ `W_O · W_V · residual[current_pos]`
4. **LoopBench v3 baseline**：1.7B=22.71%, 4B=21.43%（可用于筛查）
5. **LoopBench 严格单调不适合做主发现器**：floor effect + 漏掉 L0H3

*Moderate evidence：*

6. **0.6B 存在功能类似的 analogous L0H3**（rep delta +0.93），尚无 weight-space alignment 验证
7. **4B 在当前筛选方法下未找到 clean single-head RSH**：不等于"4B 没有"

*Working hypothesis：*

8. **L0H3 的作用机制可能是间接的**：改变 token 竞争格局（描述性标签，待 causal mediation 验证）

## 方法论教训

| 教训 | 来源 | 指导下一步 |
|------|------|-----------|
| Full head ablation sweep 是 gold standard 但 O(N) 太贵 | Phase 1 | 需要更高效的 two-stage 方法 |
| prev-1 attention rank 在小模型有效，大模型失败 | Phase 3 | 不能仅靠注意力模式筛选 |
| Scale sweep 效果依赖 benchmark | Phase 4, 6 | 需在复读倾向题上测，而非通用 benchmark |
| LoopBench 需要足够高的 baseline loop rate | Phase 6 | 用 LoopBench v3 高复读子任务子集 |
| "Efficiency head" ≠ RSH | Phase 5 | 必须以复读率变化（而非长度变化）为主指标 |

---

## Experiment 0: 全头机制评分 — Dense Mechanistic Landscape

### 方法论来源

Olsson et al. (2022) 通过为 induction head 定义两个 per-head 评分函数（prefix-matching score + copying score），在不做任何 intervention 的情况下高效定位候选 head。我们将这一方法论迁移到 RSH 研究：定义三个 RSH-specific 的评分函数，在单次 forward pass 中为全部 head 生成 dense mechanistic landscape。

### 目标

1. 为每个模型的**全部 head** 计算三个机制评分，得到非干预式的 RSH 候选排名
2. 在 1.7B 上以 gold-standard ablation 结果校准评分函数的检测性能
3. 提供跨模型可比的 dense landscape，即使某模型没有 clean single-head RSH 也能揭示功能分布模式

### 设计

**三个评分函数：**

**0a: Direct Logit Attribution (DLA)**

对每个 head h、每个 position t，计算该 head 对 repeat token logit 的贡献：

```
DLA_h(t) = head_output_h(t) · (W_unembed[:, tok_repeat(t)] - W_unembed[:, tok_best_alt(t)])
```

**Operational definitions（统一适用于本计划所有实验）：**
- `tok_repeat(t)` := position t-1 实际生成的 token（在 baseline trajectory 上）。直觉：如果模型在 position t 复读，它会再次输出 t-1 的 token
- `tok_best_alt(t)` := 在 position t 的 baseline logit 分布中，排除 `tok_repeat(t)` 后 logit 最高的 token。Tie-breaking：取 token id 最小者
- DLA 衡量的是 repeat-vs-alternative **logit margin** 的 per-head 贡献，而非绝对 logit

**两种计算模式（解决 ablation 依赖问题）：**

- **Mode A (ablation-independent, primary)**：在 **baseline trajectory 的所有 position** 上计算 DLA，不需要任何 ablation 信息。按 position 类型聚合：
  - "repeat-prone positions"：baseline 下 `tok_repeat(t)` 在 logit top-3 内的 position（即模型本身就在"犹豫"是否复读）
  - "non-repeat positions"：其余 position
  - RSH 预期特征：在 repeat-prone positions 有一致的**负 DLA**
- **Mode B (ablation-informed, supplementary)**：在 collapse-boundary positions（baseline 正常但 zero-ablation 后首次 loop 的 position ± 5 tokens）上额外统计，作为交叉验证

- 注意：DLA 是线性近似（忽略 LayerNorm 和后续 layer 的非线性）。如果 L0H3 的真实机制是间接的（通过 downstream mediation），DLA 可能较弱——这本身是信息性的，支持间接机制假说，而非方法失败

**0b: Self-Attention Concentration (SAC)**

```
SAC_h = mean_over_positions(attn_h[t, t])
```

- L0H3 已知 SAC=0.986
- 跨模型比较时，高 SAC + 负 DLA 的 head 是 RSH 候选的强信号

**0c: Downstream Composition Score (DCS)**

量化 head h 的 output subspace 与下游 head j 的 key subspace 重叠：

```
DCS_K(h→j) = ‖W_K_j · W_O_h‖_F / (‖W_K_j‖_F · ‖W_O_h‖_F)
DCS_V(h→j) = ‖W_V_j · W_O_h‖_F / (‖W_V_j‖_F · ‖W_O_h‖_F)
```

- 对 L0H3，计算其与 Layer 1–5 所有 head 的 DCS，标识 circuit 中的**候选下游边**（candidate edge prior）
- 高 DCS-K 表示下游 head 的 key subspace 与 L0H3 的 output subspace 有几何重叠，是 K-composition 的**候选**（不等于功能性连接，需 path patching 确认）
- 高 DCS-V 表示下游 head 的 value subspace 与 L0H3 output 有重叠，同样是候选而非确认

### 数据

- LoopBench v3 的 square_root 50 题（DLA 和 SAC 需要 forward pass）
- DCS 只需 weight matrices，无需 data

### 1.7B 校准（解决单正例问题）

**问题**：1.7B 全量消融中仅 L0H3 为 positive（N+=1），传统 ROC/AUPRC 在 N+=1 时不稳定，threshold 迁移不可靠。

**解决方案：rank-based 评估 + 跨模型联合校准**

1. **Rank-based metrics（主要）**：
   - Recall@k：L0H3 是否出现在 DLA/SAC 的 top-k 中？（k = 1, 3, 5, 10）
   - Rank percentile：L0H3 在 448 heads 中的 DLA 排名百分位
   - 这些指标对 N+=1 是鲁棒的（不需要 threshold）

2. **跨模型联合校准（如果 0.6B 结果可用）**：
   - 在 0.6B 上也计算全头 DLA，以 0.6B L0H3 为第二个 positive（N+=2）
   - 报告两个模型的 DLA rank 是否一致
   - 如果一致 → DLA 作为 heuristic 筛选器有一定泛化性

3. **ROC/AUPRC（supplementary，明确标注局限）**：
   - 仍然计算 ROC AUC，但标注"N+=1，仅供参考"
   - 不做 threshold 迁移；大模型上直接使用 rank-based 筛选（DLA top-10 → 进入 ablation 验证）

### 预估资源

- 0a+0b: ~0.5h/model（单次 forward pass with hooks）
- 0c: ~0.1h/model（纯 weight matrix 计算）
- 校准分析: ~0.2h

### 成功标准

- **Primary**：L0H3 在 DLA (Mode A, ablation-independent) ranking 中 rank ≤ 10（448 heads 的 top 2.2%）
- **Strong**：L0H3 DLA rank ≤ 5 且 0.6B L0H3 的 DLA rank 也 ≤ 10
- **Informative failure**：如果 DLA rank > 10 → 说明线性近似不足以捕捉间接机制，DLA 仅用于 landscape 比较（Exp 5a），不用于发现
- DCS 能标识出至少 1 个与 L0H3 DCS-K > 全头 DCS-K 的 95th percentile 的下游 head（作为 circuit 追踪起点）

---

## Experiment 1: 跨模型 RSH 定位 — Three-Stage Protocol

### 目标

在 Qwen3-4B / 7B / 8B 上定位 RSH 候选，同时校准发现方法的召回率。

### 设计

**Stage 0: 全层哨兵式粗扫（~2h/model）— 新增**

*目的*：打破 "Layer 0 prior"，避免正例只有 0.6B/1.7B 两个小模型而 4B 已给出反例的归纳风险。

*检出力设计*：假设某层有 1 个 RSH（占该层 head 总数的 1/H_per_layer），每层抽 k 个 head 时漏掉该 RSH 的概率为 (1 - 1/H_per_layer)^k。对于 4B（每层 36 heads），k=4 时单轮命中概率 ≈ 10.6%，做 R=3 轮独立随机抽样后总命中概率 ≈ 1 - (1 - 0.106)^3 ≈ 28.5%。这仍不足以支撑强负结论，因此：

- 数据：LoopBench v3 的 square_root 50 题（baseline rep 最高，信号最强）
- 操作：每层每轮随机抽 k=4 个 head 做 zero-ablation，共 R=3 轮独立抽样（不重复），总计每层测 min(12, H_per_layer) 个 head
- 指标：per-sample paired Δrep（每个 prompt 上 zero vs baseline 的 rep 状态变化）
- 输出：layer-level 热力图，标记任何 layer 有 ≥1 个 head 的 mean Δrep > layer-wise 95th percentile
- **检出力声明**：Stage 0 的负结果（某层无信号）只意味着"该层在当前覆盖率下未发现信号"，**不可作为该层无 RSH 的排除证据**。只有 Stage 1 的全 head 扫描才能支撑排除结论
- **Deeper-layer discovery hole 的缓解**：Stage 0 单层命中概率仅 ~28.5%（4B），因此 deeper-layer single-head RSH 有高概率被漏掉。为缓解此问题：
  - **Exp 0 DLA (Mode A) 提供层无关的补充信号**：DLA 在全部 head（所有层）上计算，不受 Stage 0 覆盖率限制。如果某个 deeper-layer head 在 DLA top-10 但 Stage 0 未覆盖其层 → 该层自动纳入 Stage 1
  - 如果 Exp 0 DLA 也未发现 deeper-layer 候选（DLA calibration 失败 OR 所有 top-10 均在 L0）→ 论文中明确声明"deeper-layer RSH 的排除受限于 Stage 0 覆盖率和 DLA 线性近似"
- 如果信号集中在 Layer 0 → 进入 Stage 1 only on L0
- 如果其他 layer 有信号（Stage 0 OR DLA top-10）→ Stage 1 扩展到该 layer
- 如果全部无信号 → 仍进入 Stage 1 on L0（作为 baseline），同时触发 Exp 2

**Stage 1: 目标层精扫（~2h/model）**

在 Stage 0 确定的目标层（默认 Layer 0 + 任何有信号的层）上，做全 head zero-ablation：

- 数据：LoopBench v3 中 square_root + newtons_iteration 各 50 题 = 100 题
- 操作：目标层每个 head 做 zero-ablation generation
- 指标：**prompt-paired effect estimate**（不只看 aggregate rep rate，而是对每个 prompt 计算 paired Δ）
- 统计流程（统一规范，适用于本计划所有实验）：
  1. 对每个 head，计算 prompt-level paired Δrep（binary: 0→1 / 1→0 / no change）
  2. 用 prompt-level resampling bootstrap（B=10000 次）估计 mean Δrep 的单侧 95% CI
  3. 对本 stage **全部被测 head**（不只是"通过 CI 的"）统一计算单侧 p-value（H0: mean Δrep ≤ 0）
  4. 在全部 p-value 上做 Benjamini-Hochberg FDR 校正，α=0.05
  5. 报告每个 head 的：mean Δrep, 95% CI, raw p, BH q-value, Cohen's d
  6. 筛选：BH q < 0.05 的 head 进入 Stage 2
- **Family 定义**：每个模型、每个 stage 的全部 head 构成一个 testing family
- 对照：全层其余 head 的 effect size 分布自动构成 empirical null

**Stage 2: 精筛 + 验证（~3h/model）**

对 Stage 1 候选做 scale sweep + 多 benchmark 验证：

- Scale: {0, 0.5, 1.0, 1.5, 2.0}
- 数据集: LoopBench v3 full (700), NuminaMath 100 (准确率对照)
- 指标: 复读率 + 准确率 + 生成长度
- RSH 标准: ablation 显著增加复读率（prompt-paired bootstrap p<0.05）AND scale>1 不显著降低准确率

### 1.7B 校准 (validation + rank-based evaluation)

在跑新模型之前，先在 1.7B 上跑完整 protocol，提供 ground-truth 校准：

**校准目标：**
1. **召回验证**：Stage 0 + Stage 1 能否正确召回 L0H3（要求 BH q < 0.05 且 rank ≤ 3，ranking key = BH q-value ascending）
2. **Rank-based 定量校准**：以 1.7B 全量 448-head 消融的结果为 gold label（L0H3 = positive），评估以下 proxy 方法的 recall@k（k=1,3,5,10）：
   - Exp 0 DLA rank (Mode A, ablation-independent)
   - prev-1 attention rank
   - wait-token logit delta rank
   - Stage 1 bootstrap q-value rank
   - 输出：各方法的 recall@k 对比表 + L0H3 在各方法中的 rank percentile
   - *注意*：由于 N+=1，不做 full ROC/AUPRC threshold 迁移。大模型上使用统一的 BH q < 0.05 或 DLA top-10 → ablation 验证 的两步流程
3. **大模型筛选策略**：Stage 1 的 BH q < 0.05 作为主筛选标准；DLA top-10 作为辅助参考（如果 Exp 0 校准显示 DLA recall@10 ≥ 1）

**如果 L0H3 被漏掉**：说明 protocol 检出力不足，需在推广前调整（增加 sample size 或降低 Stage 0 粒度）。

### 预估资源

| Model | Stage 0 | Stage 1 | Stage 2 (est.) | Total |
|-------|---------|---------|----------------|-------|
| 1.7B (校准) | ~0.5h | ~1h | — | ~1.5h |
| 4B | ~1.5h | ~3h | ~3h | ~7.5h |
| 7B | ~2h | ~4h | ~4h | ~10h |
| 8B | ~2h | ~4h | ~4h | ~10h |

### 脚本

复用 `scripts/run_local_heads_repetition_screen.py` + `scripts/run_loopbench_baseline_repetition.py`，需适配：
- 支持指定 layer 范围（而非只跑 local heads）
- 支持 high-loop 子任务子集
- 支持 per-layer random sampling mode (Stage 0)
- 输出 prompt-paired effect estimates + bootstrap CI

### 成功标准

- 1.7B 校准：L0H3 的 BH q < 0.05 且 rank ≤ 3（by q-value ascending）；ROC/AUPRC 报告完成
- 至少 1 个 ≥4B 模型找到 BH q < 0.05 的 RSH 候选
- 或确认 ≥4B 模型在 Stage 1 扫描范围内无 q < 0.05 的 head → 触发 Exp 2（注意：Stage 0 负结果不构成排除证据）

---

## Experiment 2: 分布式抑制假说测试（条件触发）

### 触发条件（修订版 — CI-based，支持弱信号模式）

满足以下任一条件时触发：

1. **无信号**：Exp 1 Stage 0+1 在 4B 所有扫描层中，无 head 的 bootstrap CI lower bound > 0
2. **多头弱信号**：存在 ≥3 个 head 的 mean Δrep 在 +0.03 ~ +0.08 之间但 CI 跨零 — 这提示分布式抑制，应该用 grouped ablation 而非单头扫描来测试
3. **噪声假阳性**：有 head 通过 CI 筛选但在 Stage 2 的 scale sweep 中不复现

*注意*：一个 head 恰好 Δrep=+0.11 不应阻止测试分布式假说；它可能是噪声。

### 目标

区分两种可能：(a) 4B 的 RSH 在 deeper layer（单头扫描没覆盖到），(b) 复读抑制已分布式化（无单头 RSH）。

### 设计

**2a: 全层单头精扫（if Stage 0 有 deeper layer 信号）**

- 对 Stage 0 标记的 layer 做全 head zero-ablation
- 与 Exp 1 Stage 1 相同的 prompt-paired + bootstrap protocol

**2b: Grouped ablation（if 多头弱信号或全无信号）**

- **Selection**: 把 Stage 0/1 中 mean Δrep 排名前 k 的 head 分组同时消融
- **防 winner's curse**: top-k 选择基于 Stage 1 的 discovery set（100 题），验证使用 **held-out prompt set**（LoopBench v3 的 long_division + logical_paradox 各 25 题 = 50 题，与 Stage 1 无 prompt 重叠）
- **对照设计**（3 种，防止"同层/同功能相关性"假阳性）：
  - C1: 随机 k-head group × 20 次 permutation（基线）
  - C2: 同层 matched random k-head group × 20 次（控制 layer 效应）
  - C3: 按 attention locality 类型 matched random k-head group × 20 次（控制功能类别）
- **剂量曲线**: 对 k = {2, 3, 5, 8} 做 grouped ablation，观察 group Δrep 是否随 k 单调增长
- **Additivity/synergy 检验**: 比较 group-k 的 Δrep 与 k 个单头 Δrep 之和。如果 group > sum → synergy（head 间有冗余/互补）；如果 group ≈ sum → additive（独立效应）；如果 group < sum → self-repair/compensation
- **Claim 限制**: group effect 显著 → 只能说"这组 head 联合消融后复读上升"，不能直接等价于"分布式冗余假说成立"。后者还需要 additivity 分析和 held-out 复现

**2c: Leave-k-out（if 2b 显示分布式信号）**

- 从 top-k group 中逐个移除 head，观察 Δrep 的边际贡献
- 建立 head 重要性排序
- 同样在 held-out prompt set 上验证

### 预估资源

- 2a: ~8h (仅扫标记层)
- 2b: ~3h (5 head × 100 samples × 16 permutations)
- 2c: ~2h (5 条件 × 100 samples)

### 成功标准

- 2a 找到 BH q < 0.05 的 head → 进入 Exp 1 Stage 2
- 2b 的 top-k group Δrep 在 held-out set 上显著大于 3 种 matched controls (p<0.01, Bonferroni 校正)，且 additivity 分析显示 synergy 或 additive → **初步证据支持多头联合抑制**（需要在论文中限定为 "these heads jointly contribute to suppression" 而非 "distributed redundancy"）
- 2b 无差异或 held-out 不复现 → **结论：attention head 层面可能不是 4B 复读抑制的主要承载层，建议转向 MLP/SAE 路线**

---

## Experiment 3: L0H3 机制 — Causal Mediation Analysis

### 目标

通过因果中介分析确定 L0H3 抑制复读的具体路径，判别 primary hypothesis **H-attn**（downstream attention redistribution）vs competitor **H-mlp**（MLP gate shift），并进一步分解 L0H3 output 的 norm vs direction 贡献。

### 设计

**Phase A: 描述性分析（全样本 + 分层）**

在 **全部 100 个样本**上（不仅限于 ablation 后进入复读的样本）记录 baseline 和 zero(L0H3) 的差异，**然后**按 outcome 分层：

- **全样本分析**（避免 selection bias / outcome conditioning）：
  - 每个 position 的 residual stream norm（Layer 0 output 之后）
  - L0H3 output vector norm 占 residual stream 总 norm 的比例
  - L0H3 output vector 与 residual stream 的 cosine similarity
  - Layer 1–5 所有 head 的 attention entropy
  - Post-Layer-0 LayerNorm scale 的变化

- **分层对比**：
  - Group A: ablation 后进入复读的样本（~50 个）
  - Group B: ablation 后仍正常的样本（~50 个）
  - 比较两组在上述指标上的分布差异

**Phase B: Circuit Verification + 因果中介实验（核心）**

借鉴 Induction Heads 的方法论：先用 direct logit attribution 分解模型行为到 per-component 贡献，再用 patching 验证因果路径。

在 **全部 100 个样本**上做以下 intervention（不仅限于 ablation-induced loop samples），然后按 outcome 分层分析。这避免 conditional mechanism analysis 的 selection bias，同时保留对 loop-causing subset 的深入分析。

**Patch locus 精确定义**（统一适用于 3b-1 ~ 3b-3）：
- **Tensor**: Layer 0 attention output 之后、Layer 0 MLP 之前的 residual stream（即 `x + attn_out`）
- **Position**: 每个生成 token 的 position（逐 token intervention，不是整个 sequence 一次性 patch）
- **Pre/Post-LN**: intervention 作用在 **post-attention, pre-MLP** 的 residual stream 上（即 Layer 0 的 attention sublayer 输出之后，进入 Layer 0 MLP 的 LayerNorm 之前）

**3b-0: Per-component Logit Attribution（全景分解，类比 Induction Heads）**

在 collapse-boundary positions 上，将 repeat-token logit 分解为每个组件的贡献：
- 对 **每个 attention head**（全层）和 **每个 MLP layer** 的 output，投影到 `W_unembed[:, tok_repeat] - W_unembed[:, tok_best_alt]` 方向
- 这给出 **repeat-vs-alternative logit diff 的全景分解**：哪些组件在促进 repeat，哪些在抑制？
- 在 baseline 和 zero(L0H3) 两个条件下都计算，观察 L0H3 ablation 后哪些组件的贡献发生了最大变化
- 这直接回答"L0H3 的效应经由哪些下游组件传递"（被影响最大的组件 = circuit 成员）

**3b-1: Norm-preserving direction patch（测试方向贡献）**
- 在每个生成 position，取 zero(L0H3) 后的 residual stream r_zero，rescale 为 r_zero × (‖r_baseline‖ / ‖r_zero‖)
- 保留 zero(L0H3) 的方向，恢复 baseline 的 norm
- 如果复读消失 → norm 是关键因子（L0H3 主要通过维持 residual norm 起作用）
- 如果复读仍在 → norm 不是充分因子

**3b-2: Direction-preserving norm patch（测试 norm 贡献）**
- 在每个生成 position，取 zero(L0H3) 后的 residual stream r_zero，旋转到 baseline 方向：r_patched = (r_baseline / ‖r_baseline‖) × ‖r_zero‖
- 保留 zero(L0H3) 的 norm，恢复 baseline 的方向
- 如果复读消失 → 方向是关键因子（L0H3 主要通过引入方向偏移起作用，支持 H-attn 的 Step 2 假设）
- 如果复读仍在 → 方向不是充分因子

**3b-3: Project-out L0H3 direction（测试效应集中度）**
- 从 baseline residual 中仅 project out L0H3 mean output direction（从 Phase A 的全样本平均得到）：r_patched = r_baseline - (r_baseline · d̂) × d̂
- 比较 project-out 与 zero-ablation 的 Δrep：如果一致 → 效应集中在该方向；如果 project-out 弱于 zero-ablation → L0H3 的效应不仅是一个固定方向

**3b-4: Path patching 到 repeat-logit（测试中介路径）**
- 在 collapse boundary position（定义：baseline 正常但 zero(L0H3) 首次进入 loop 的 position ± 5 tokens），对两条路径做 activation patching：
  - Path A: L0H3 output → Layer 1-5 attention queries/keys（经 residual stream 间接传递）
  - Path B: L0H3 output → Layer 0 MLP input（经 LayerNorm 直接传递）
- 度量：`logit_margin(t) = logit(tok_repeat(t)) - logit(tok_best_alt(t))`（与 Exp 0 DLA 定义一致）
- **Mediation % 计算公式**：
  ```
  total_effect = logit_margin_zero(t) - logit_margin_baseline(t)   [正值：ablation 增加 repeat margin]
  restored_A   = logit_margin_zero(t) - logit_margin_patch_A(t)    [正值：恢复 Path A 后 repeat margin 降低了多少]
  mediation_A  = restored_A / total_effect × 100%
  ```
  其中 `patch_A` 表示：从 zero(L0H3) 状态出发，仅将 Path A（Layer 1-5 attn 的 K/Q 输入）替换为 baseline 值。如果 Path A 承载了 L0H3 的全部效应，`patch_A` 的 logit margin 应恢复到 baseline 水平，此时 `mediation_A = 100%`。如果 Path A 不承载效应，`patch_A ≈ zero`，此时 `mediation_A ≈ 0%`。mediation_B 类似。edge case：如果 `|total_effect| < 0.1`，该 position 排除
- **判别标准**：mediation_A > 60% → 采纳 H-attn；mediation_B > 60% → 采纳 H-mlp；两者均 30-60% → mixed circuit
- 确定 L0H3 的因果效应主要经由哪条下游路径
- **与 Exp 0 DCS 的交叉验证**：3b-4 的 path patching 结果应与 Exp 0c 的 downstream composition score 一致——DCS 高的下游 head 应该在 path patching 中也显示高 mediation（DCS 是 candidate edge prior，path patching 是功能确认）

**3b-5: Circuit Diagram Synthesis**
- 汇总 3b-0（per-component attribution）、3b-4（path patching）、Exp 0c（DCS）的结果
- 输出：一个可视化的 RSH circuit diagram，标注每个组件的贡献和路径权重
- 标准：circuit 中的组件贡献之和应解释 ≥80% 的 L0H3 ablation 导致的 repeat logit diff 变化

**允许两层 mixed 结果**：
- **Mixed contribution**（3b-1/3b-2）：norm 和 direction 可能同时贡献。recovery rate 均在 30-60% → 报告为 mixed contribution
- **Mixed circuit**（3b-4）：H-attn 和 H-mlp 路径可能同时成立。mediation 均在 30-60% → 报告为 mixed circuit
两者独立判定，不强制二选一。

### 对照

- 10 个随机 Layer 0 head 做同样的 intervention suite 作为 null distribution
- Group B 样本（ablation 后不 loop）验证 intervention 效果的特异性

### 脚本

`scripts/analyze_residual_at_collapse.py` 已存在，需大幅扩展：
- 全样本 + 分层分析框架
- Norm-preserving / direction-preserving patch
- Project-out intervention
- Path patching module（可复用 `cot_research/head_intervention.py` 的 hook 机制）

### 预估资源

- Phase A: ~2h（100 samples × 2 conditions × hooks）
- Phase B: ~4h（50 samples × 4 intervention types × 11 heads）
- 总计: ~6h GPU

### 成功标准

- **Norm vs Direction 判别**：3b-1 和 3b-2 的 recovery rate 差异显著（paired bootstrap p<0.01）→ norm 或 direction 的主导贡献确定
- **Mixed contribution**：如果两者 recovery rate 都在 30–60% 之间且差异不显著（p > 0.01）→ 报告为 mixed contribution（norm 和 direction 均有贡献），给出各自比例
- **H-attn vs H-mlp 判别**：3b-4 的 path patching 中 Path A (downstream attn) 或 Path B (MLP0) 的 mediation > 60% → 采纳对应 primary/competitor hypothesis。两者均 30-60% → mixed circuit
- **Null baseline**：以上所有效果在 10 个 random head 的 null distribution 中不出现（permutation p<0.01）
- **全样本 vs 分层**：Phase B 的全样本分析结果与 loop-causing subset 方向一致（排除 outcome conditioning artifact）

---

## Experiment 4: L0H3 Output Vector 功能表征 + OV Circuit 深入分析

### 目标

理解 L0H3 output（≈ W_O · W_V · residual[current_pos]）在 vocab space 中的语义，并与 Exp 0 的 DLA 和 Exp 3b-0 的 per-component attribution 交叉验证。

### 设计

**4a: Output Vector → Logit Space（与 DLA 交叉验证）**
- 对 100 个样本，提取每个 position 的 L0H3 output vector
- 通过 unembedding matrix 映射到 vocab space
- **核心指标**：repeat token vs best alternative 的 logit diff（即 Exp 0 中 DLA 的 sample-level 细化版）
- 分析 top suppressed / boosted tokens 与 position context 的关系
- **与 DLA 的一致性检查**：4a 的 per-position logit diff 应与 Exp 0 的 DLA 数值一致（两者计算方式相同，4a 更细粒度）
- *注意*：直接 unembed 是线性近似（忽略后续 LN/attn/MLP），结论需与 Exp 3b-0 的全景分解和 3b-4 的 path patching 交叉验证

**4b: Position Dependency + OV Eigenspectrum**
- L0H3 output 在不同 position（early/mid/late/collapse-boundary）是否变化？
- 如果高度一致（PCA PC1 explained variance > 50%）→ 固定 bias
- 如果 position-dependent → 需要理解 LayerNorm 在其中的作用
- **OV eigenspectrum 分析**（类比 Induction Heads 的 OV matrix 分析）：计算 W_OV = W_V · W_O 的 top eigenvalues/eigenvectors，判断 OV circuit 是近似 identity（copying）还是特定方向投影（bias）

**4c: 跨样本一致性**
- Pairwise cosine similarity of L0H3 output vectors across 100 samples
- 与 random Layer 0 heads 对比
- **与 circuit hypothesis 的关联**：如果 output vector 高度一致（cosine > 0.8）→ 支持"fixed bias"假说（circuit Step 2）；如果 context-dependent → 需修正 circuit hypothesis

### 脚本

`scripts/characterize_l0h3_output_vector.py` 已存在

### 预估资源

~1.5h GPU

### 成功标准

- 明确 L0H3 是 fixed bias 还是 context-dependent function
- 如果是 fixed bias：它 suppress 了哪些 token？是否包含 loop-prone tokens？

---

## Experiment 5: 跨模型 RSH 机制比较（Common Scoring Framework）

### 触发条件

Exp 0 的 DLA landscape 完成 OR Exp 1 在 ≥1 个新模型上找到 RSH 候选。

### 目标

通过统一的评分框架比较不同模型的 repeat-suppression 机制，回答"circuit 的哪个组件随规模变化了？"

### 设计

**5a: DLA Landscape 跨模型比较（不依赖 RSH 候选发现）**
- 对每个模型，使用 Exp 0 的 DLA landscape（全部 head 的 repeat-suppression logit 贡献）
- 比较：
  - DLA 的 **集中度**：top-1 head 的 DLA 占全部负 DLA 之和的比例 → 集中度随规模下降 = focal→distributed 的直接证据
  - DLA 的 **层分布**：负 DLA 主要集中在哪些 layer？Layer 0 的份额是否随规模下降？
  - SAC-DLA **联合分布**：是否只有高 SAC 的 head 才有显著负 DLA？
- 这一分析**不需要**找到 clean single-head RSH，只需要 DLA landscape

**5b: RSH Circuit 比较（依赖 RSH 候选发现）**
- 对每个确认的 RSH，重复 Exp 3 和 Exp 4 的核心分析：
  - Self-attention mass (SAC)
  - DLA (repeat-suppression 贡献)
  - Residual stream norm/direction effect
  - OV eigenspectrum
  - DCS (downstream composition) → 是否与同层/同功能的下游 head 组成 circuit？
- 比较各模型 RSH 的 circuit diagram（来自 3b-5）是否在结构上保守

### 成功标准

- 5a: DLA 集中度指标显示跨规模的系统性变化（单调下降或分段变化），且 trend 在 ≥3 个模型上一致
- 5b: RSH 在不同模型上共享 ≥2 个 circuit 特征（SAC, DLA sign, DCS pattern） → 保守 circuit
- 5b: circuit 特征不共享 → 不同模型采用不同策略抑制复读

---

## Experiment 6: RSH 特异性与解码稳健性验证

### 目标

(a) 验证 RSH 效果是否特异于"病态复读"；(b) 验证效果在不同解码配置下是否稳定。

### 设计

**6a: 特异性测试**
- 构建"正常重复"数据集：数学公式重现、合理的步骤回顾等（~50 samples）
- 对 RSH 做 zero-ablation 和 scale=1.5
- 指标：这些样本的生成质量是否受影响

**6b: 解码稳健性测试**
- 在 LoopBench high-loop subset 上，对 L0H3 做 zero-ablation
- 解码配置：greedy (temp=0) / sampling (temp=0.6, top-k=50) / nucleus (temp=1.0, top-p=0.95)
- 对比三种配置下 ablation 的 Δrep 是否一致
- 如果 greedy 下效应强但 sampling 下消失 → 效应可能是解码伪影

### 预估资源

- 6a: ~2h
- 6b: ~3h (3 configs × 100 samples × 2 conditions)

### 成功标准

- 6a: RSH ablation 对正常重复样本无显著影响 → 特异性确认
- 6b: Δrep 在 ≥2 种解码配置下方向一致且 CI 不跨零 → 解码稳健性确认
- 如果 6b 失败 → 需重新评估所有基于 greedy decoding 的结论

---

## 实验优先级与依赖关系

```
Exp 0 (全头机制评分, 1.7B 校准)
        │
        ├──→ Exp 0 (4B/7B/8B)  ──→  Exp 5a (DLA landscape 跨模型比较)
        │
        └──→ (DLA rank 辅助 Exp 1 候选排序)

Exp 1 Stage 0 (全层哨兵)
        │
        ├──→ Exp 1 Stage 1 (目标层精扫)  ──→  Exp 1 Stage 2 (验证)
        │                                            │
        │                                            ├──→ Exp 5b (RSH circuit 比较)
        │
        └──→ Exp 2 (分布式假说测试, 条件触发)
                    ├── 2a: deeper layer 精扫
                    ├── 2b: grouped ablation
                    └── 2c: leave-k-out

Exp 3 Phase A (全样本描述)
        │
        ├── Exp 4 (output characterization + OV eigenspectrum)
        │
        └──→ Exp 3 Phase B (circuit verification + causal mediation)
                    ├── 3b-0: per-component logit attribution
                    ├── 3b-1/2: norm vs direction decomposition
                    ├── 3b-3: project-out
                    ├── 3b-4: path patching (与 Exp 0 DCS 交叉验证)
                    ├── 3b-5: circuit diagram synthesis
                    │
                    └──→ Exp 6 (特异性验证)
```

**Phase 7 建议执行顺序：**

1. **Exp 0 (1.7B 校准)** + **Exp 1 Stage 0 (1.7B 校准)**：~1h，并行执行。Exp 0 提供 DLA landscape，Exp 1 验证 ablation protocol
2. **并行启动 Exp 0 (4B/7B/8B) + Exp 1 (4B/7B/8B Stage 0+1) + Exp 3 Phase A + Exp 4**（互不依赖，总计 ~15h GPU）
3. Exp 3 Phase A 完成后启动 **Exp 3 Phase B**（~5h，含 3b-0 per-component attribution）
4. 根据 Exp 1 结果决定是否触发 **Exp 2**（条件分支）
5. Exp 0 全模型完成后启动 **Exp 5a**（DLA landscape 比较，不等 RSH 发现）
6. Exp 3 Phase B 完成后启动 **Exp 6**
7. 新模型 RSH 确认后启动 **Exp 5b**（circuit 比较）

### 关键 confound 清单

| Confound | 风险 | 缓解措施 |
|----------|------|---------|
| Zero-ablation 是 off-manifold 扰动 | 中高 | Exp 3b-3 的 project-out 作为更温和的替代；Exp 6 的解码稳健性测试 |
| 生成长度与 loop 检测强耦合 | 中 | 区分 "真正恢复正常" vs "打满 max_tokens 但未触发 loop 规则" |
| 仅用 high-loop subset 可能找到 task-specific suppressor | 中 | Exp 1 Stage 2 在 full 700 题上验证；Exp 6 跨题型测试 |
| 不同模型 head 数不同，单头 Δrep 有规模 confound | 低中 | 在 bootstrap CI 中使用 per-prompt paired design，不直接跨模型比较绝对值 |
| Temperature / decoding config 伪影 | 中 | Exp 6 增加 decoding robustness sweep (greedy / temp=0.6 / temp=1.0) |

---

## 基础设施

- **Remote server**: `ssh -p 8002 yucheng@101.6.96.183`, GPUs 0-8
- **Environment**: `micromamba activate qwen_math`
- **Code base**: `cot_research/` (framework) + `scripts/` (experiment scripts)
- **Data**: `evaluation/data/loopbench_reconstructed_v2/test.jsonl` (700 题)
- **Results**: `experiment_results/experiments/phase7_*/`

## Target Venue

ICLR 2027 或 ACL 2026 ARR December
