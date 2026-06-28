# Idea Report: 复读抑制机制的跨规模演变 — 从单头 RSH 到分布式抑制

**Direction**: 在 Qwen3 系列 ≤10B 模型中追踪复读抑制机制随规模的演变：小模型的 focal single-head RSH → 大模型的分布式抑制  
**Date**: 2026-04-29（基于 Phase 1–6 实验结果全面更新）  
**Pipeline**: 6 轮实验 → 方法论反思 → mechanistic scaling story  
**Models**: Qwen3-0.6B / 1.7B / 4B (已测); 7B / 8B (待测)

---

## 1. Executive Summary

经过 6 个阶段、32+ 个实验，我们发现了一个 mechanistic scaling pattern：小规模 Qwen3 模型（0.6B/1.7B）中存在一个保守的 Layer-0 self-attending suppressor head（L0H3），它不是经典的 copy suppression head，而是通过改变 residual stream 中的 token 竞争格局来稳定解码轨迹；但到 4B，这种单头可定位性显著下降，提示复读抑制机制随规模从 focal 走向 distributed。

**分层结论（按证据强度排序）：**

*Strong evidence（多实验交叉验证）：*

1. **1.7B L0H3 是当前实验条件下观察到的唯一单头因果必要因子**：在 greedy decoding + zero-ablation 设置下，全量 448-head 消融中仅 L0H3 被屏蔽后稳定引发复读（rep rate 0%→99%）。*Scope: 特定 prompt set / decoding config / intervention type。*
2. **L0H3 不符合经典 copy suppression 定义**：OV circuit 的静态线性近似下 top-10 suppression=0%。*注意：这只排除了明显的、静态的 top-rank OV 抑制，不排除低秩、context-conditional、经 LN/MLP 放大的间接 logit 效应。*
3. **LoopBench 严格单调筛选不适合作为 RSH 主发现器**：baseline loop rate 过低导致 floor effect，scale=1.5 不产生 clean suppressor。

*Moderate evidence（需进一步验证）：*

4. **0.6B 存在功能类似的 analogous head**（同为 L0H3，rep delta +0.93）。*尚无 weight-space alignment 或 head correspondence 验证，不宜称"同源"。*
5. **4B 在当前筛选方法下未找到 clean single-head RSH**：L0H2（排名第一）ablation 无复读效应，L0H22 效果弱，L0H9/L15H10 仅为 efficiency head。*这不等于"4B 没有 RSH"——可能是发现方法不适用，或抑制功能已分布式化。*

*Working hypothesis（待因果验证）：*

6. **L0H3 的作用机制可能是间接的**：10 个 case study 提示它通过改变 token 整体竞争格局影响复读，而非直接抑制特定候选 token。*目前是描述性标签，需要 causal mediation 实验确认具体路径。*
7. **Scale intervention 效果依赖 benchmark 和题型**：同一个 head 在不同题型上可能表现相反。

**核心开放问题：**

- L0H3 "改变 token 竞争格局" 的具体因果路径是什么？是 norm 维持、方向偏移、还是下游 MLP 门控？
- 4B 的复读抑制是分布式的（无单头 RSH），还是我们的发现方法有系统性盲区？
- 7B/8B 模型的复读抑制机制在 focal–distributed 光谱上的哪个位置？
- Zero-ablation 作为干预手段是否产生了 off-manifold artifacts？

### 1.1 替代假说表

以下替代解释尚未被充分排除，需要在后续实验中系统测试：

| 替代假说 | 描述 | 当前证据状态 | 排除所需实验 |
|---------|------|------------|------------|
| **分布式多头抑制** | 复读抑制由多个 head 冗余承担，单头 ablation 效应因 self-repair 被掩盖 | 4B 结果支持此假说 | leave-k-out grouped ablation |
| **MLP-mediated 抑制** | L0H3 的作用完全通过改变 MLP 输入分布实现，head 本身无直接 logit 效应 | 与"间接机制"假说一致，未验证 | path patching 到 MLP output |
| **Attention sink / value drain** | L0H3 的 98.6% self-attention 是 attention sink 行为，其"抑制"只是 value drain 副作用 | 已知 sink-like，未区分 | 与已知 pure-sink heads 对比分析 |
| **LayerNorm 门控** | L0H3 output 主要通过改变 post-L0 LayerNorm scale 影响后续层 | 未测试 | Exp 3 中加 LN scale 追踪 |
| **任务/题型特异 suppressor** | L0H3 只在特定题型（数值计算）上有 RSH 功能，在其他题型上无效 | Phase 4/6 部分支持 | 多题型特异性测试 (Exp 6) |
| **解码配置伪影** | ablation 效果强烈依赖 temperature/top-k/max_tokens | 未系统测试 | decoding robustness sweep |

---

## 2. 已有实验知识全景

### 2.1 跨模型 RSH 发现结果

| Model | Params | Heads | RSH Candidate | Discovery Method | Ablation Effect | Verdict |
|-------|--------|-------|---------------|-----------------|-----------------|---------|
| Qwen3-0.6B | 0.6B | 224 | L0H3 | prev-1 attention + wait ablation | rep delta +0.93 | **PASS** |
| Qwen3-1.7B | 1.7B | 448 | L0H3 | full head sweep | rep rate 0%→99% | **PASS (gold standard)** |
| Qwen3-4B | 4B | 576 | L0H2 (best rank) | prev-1 attention + wait ablation | rep delta 0.0 | **FAIL** |
| Qwen3-4B | 4B | 576 | L0H22 (wait-sensitive) | wait ablation only | scale effect weak | **WEAK** |
| Qwen3-4B | 4B | 576 | L0H9 (efficiency) | current-token top10 | length -190 (scale 2.0) | **EFFICIENCY ONLY** |
| Qwen3-4B | 4B | 576 | L15H10 (efficiency) | local head scale scan | length ↓, acc stable | **EFFICIENCY ONLY** |

### 2.2 L0H3 (1.7B) 已知属性

| 属性 | 值 | 来源 |
|------|-----|------|
| Self-attention mass | 98.6% | Phase 3: attention_pattern |
| Zero-ablation → copy rate | 46% → 99% | Phase 1: wait_head_ablation |
| Zero-ablation → rep rate | 0% → 99% | Phase 1: wait_head_ablation |
| Zero-ablation → length delta | +36 tokens | Phase 1: wait_head_ablation |
| Scale 1.5 → length delta | -274 tokens (NuminaMath) | Phase 1: scale_reflection |
| OV top-10 suppression | 0% | Phase 3: ov_circuit |
| Copy suppression | 否定 | Phase 3: copy_suppression |
| Mechanism | 间接改变 token 竞争格局 | Phase 3: repetition_causal |
| Collapse rescue (scale=1.5) | 108 cases rescued | Phase 3: collapse_prefix |

### 2.3 发现方法论评估

| 方法 | 描述 | 优点 | 盲区 | Phase | 1.7B 上的召回/精度 |
|------|------|------|------|-------|-------------------|
| Full head ablation sweep | 逐个 zero-ablation 448 heads | 无遗漏，gold standard | O(N) 太贵，>10B 不可行 | 1 | recall=1.0 (by definition) |
| Wait-token logit ablation | 看 wait token logit delta | 快速筛查 | 与 RSH 概念有偏差，漏掉非 wait-related RSH | 1 | recall 未标定 |
| prev-1 attention + wait 联合 | 结构+功能联合排名 | 1.7B/0.6B 有效 | 4B 失败：排名第一的 L0H2 无复读效应 | 3 | L0H3 rank=1 (precision TBD) |
| Local head 分类 + scale sweep | 先按 locality 粗筛，再测 scale | 缩小搜索空间 | 假设 RSH 是 local head，可能错误 | 5 | L0H3 in local set ✓, 但 strict filter 漏掉 |
| LoopBench 严格单调 | scale=0/1/1.5 复读率单调下降 | 高精度 | **极低召回**：baseline 1/100 loop，漏掉 L0H3 | 5, 6 | recall=0 for L0H3 |
| LoopBench v3 + scale | 修复后的 LoopBench (22% baseline) | 更高 baseline | scale=1.5 无 clean suppressor；伪改善风险 | 6 | 未产生 clean candidate |

**已纳入正式 protocol**：见 EXPERIMENT_PLAN Exp 0 + Exp 1 的 "1.7B 校准" 步骤。以 1.7B 全量 448-head 消融为 gold label（N+=1），使用 rank-based metrics（recall@k, rank percentile）评估各 proxy 方法的检测性能。由于 N+=1 的局限，不做 threshold 迁移；大模型上使用 BH q < 0.05（主）+ DLA top-10（辅）的两步筛选流程。

### 2.4 未尝试但值得考虑的发现策略

| 策略 | 描述 | 预期优势 | 复杂度 |
|------|------|---------|--------|
| **Grouped ablation / coarse-to-fine** | 先按 layer 或 head cluster 分组消融，再递归细化 | 从 O(N) 降到 O(N/k + k·candidates)；可测试分布式假说 | 中 |
| **Collapse-boundary single-step screen** | 只在复读临界 prefix 上看 next-token logit diff，不跑完整 generation | 单步 forward pass，比全 generation 快 10–50× | 低 |
| **Activation / path patching** | 从 clean trajectory 和 collapse trajectory 间做因果 tracing | 比纯 ablation 更能定位真正中介路径；可区分直接 vs 间接效应 | 高 |
| **Leave-k-out / sparse regression** | 同时消融 k 个 head，用 combinatorial design 测分布式抑制 | 专门测试"多头冗余"假说，避免"找不到单头=分布式"的推论谬误 | 中高 |
| **MLP/SAE 互补路线** | 用 SAE 提取复读特征，追溯上游 head 来源 | 绕过 head-level 假设局限 | 高（需 SAE） |

### 2.5 RSH 机制评分函数（借鉴 Induction Heads 方法论）

Olsson et al. (2022) 通过定义两个 per-head 评分函数（prefix-matching score + copying score）在不做任何 intervention 的情况下高效定位 induction heads。我们为 RSH 定义三个类似的评分函数，提供与 ablation 互补的非干预证据流：

| 评分函数 | 定义 | 类比 Induction Heads | 预期 RSH 特征 |
|---------|------|---------------------|--------------|
| **Direct Logit Attribution (DLA)** | 将 head h 在 position t 的 output 投影到 repeat-vs-alternative logit margin：`DLA_h(t) = head_output_h(t) · (W_unembed[:, tok_repeat(t)] - W_unembed[:, tok_best_alt(t)])`。衡量该 head 对复读决策边界的贡献 | Copying score（但符号相反，且是 margin-based） | RSH 应在 repeat-prone positions 有**一致的负 DLA**（降低 repeat-vs-alt margin） |
| **Self-attention concentration (SAC)** | head h 对当前 position 的 attention mass：`SAC_h = attn_h[t, t]`，在所有 position 上平均 | 无直接类比 | L0H3 的 SAC=0.986；跨模型 RSH 如果保守，应有类似的高 SAC |
| **Downstream composition score (DCS_K / DCS_V)** | 量化 head h 的 output subspace 与下游 head j 的 key 或 value subspace 重叠：`DCS_K(h→j) = ‖W_K_j · W_O_h‖_F / (‖W_K_j‖_F · ‖W_O_h‖_F)`，DCS_V 类似 | K-composition（previous-token head → induction head 的路径） | 高 DCS 的下游 head 是 RSH circuit 的**候选边**（candidate edge prior），需由 path patching 确认功能性连接 |

**关键限定**：
- **DLA 是线性近似**：如果 L0H3 的真实机制主要通过 downstream mediation 起作用（间接效应），DLA 可能较弱，但这不排除 L0H3 是 RSH。DLA 校准失败是信息性的（支持间接机制假说），而非方法失败
- **DCS 衡量几何相容性，非功能性连接**：高 DCS 只表示 subspace overlap，不等价于"下游 head 读取了 L0H3 的信息"。DCS 提供 candidate edge prior，真正的 circuit edge 由 path patching（Exp 3b-4）决定

**用法**：
- **发现阶段**（Exp 0）：对全部 head 计算三个评分函数，得到 dense mechanistic landscape，无需 generation
- **校准**：在 1.7B 上以 gold-standard ablation 结果为 label，评估各评分函数的 recall@k 和 precision@k（见 Exp 0 校准设计）
- **跨模型比较**：评分函数的 landscape 提供 model-agnostic 的比较框架（Exp 5）

### 2.6 RSH Circuit Hypothesis（待验证）

借鉴 Induction Heads 的"二步 circuit"叙述方式，我们为 L0H3 提出一个 **primary circuit hypothesis** 和一个 **competitor hypothesis**，而非开放式树结构：

**Primary Hypothesis (H-attn): L0H3 → Residual Direction Shift → Downstream Attention Redistribution**

```
Step 1:  L0H3 self-attends (SAC=0.986) → 读取当前 position 的 residual stream
                ↓
Step 2:  L0H3 OV circuit 产生 direction-shifting output → 加入 residual stream
         → 改变 Layer 1+ attention heads 的 key/query 匹配
                ↓
Step 3:  Downstream attention heads 重新分配注意力 → 减少对 recent tokens 的过度关注
                ↓
Result:  最终 logit 分布中 repeat token 概率降低
```

**Competitor Hypothesis (H-mlp): L0H3 → Norm/Direction Change → MLP0 Gate Shift**

```
Step 1:  同上
Step 2:  L0H3 output 主要通过改变 residual stream 的 norm 或方向
         → 影响 Layer 0 MLP 的 LayerNorm 输入
Step 3:  MLP0 的 gating 行为改变 → 不同的 token representation 被激活
Result:  同上
```

**判别标准**：Exp 3b-4 的 path patching 将 L0H3 的效应分别归因到 Path A (downstream attn) 和 Path B (MLP0)。如果 Path A mediation > 60% → 采纳 H-attn；Path B > 60% → 采纳 H-mlp；两者均 30-60% → 报告为 mixed circuit。

**每步的可测试预测：**

| Circuit Step | 可测试预测 | 验证实验 | 当前证据 |
|-------------|-----------|---------|---------|
| **Step 1: Self-attend** | L0H3 attention mass > 95% 在 current position | Exp 0 (SAC score) | ✅ 98.6% |
| **Step 2: Output vector** | L0H3 output 在 residual stream 中引入方向偏移（cosine < 0.99 with/without L0H3） | Exp 3A + Exp 4b | ⚠️ OV top-10 suppression=0%，但可能是 context-conditional |
| **Step 3 (H-attn)** | 存在 Layer 1+ head 的 DCS-K 与 L0H3 显著高（candidate edge），且 path patching 确认 mediation | Exp 0 (DCS, candidate prior) + Exp 3b-4 (path patching, 功能确认) | **未测** |
| **Step 3 (H-mlp)** | L0H3 ablation 后 MLP0 output 方向显著改变 | Exp 3b-4 (path patching) | **未测** |
| **Result** | L0H3 ablation 后 repeat token logit 上升 | Exp 1 (ablation) | ✅ rep 0%→99% |

**与 Induction Heads circuit 的结构对比**：

| 维度 | Induction Head Circuit | RSH Circuit (H-attn, primary) |
|------|----------------------|-------------------------------|
| **步数** | 2-step (prev-token head → induction head) | 3-step (self-attend → direction shift → downstream attn redistribution) |
| **层跨度** | 跨 2+ 层 | 从 Layer 0 向下游扩展 |
| **关键组合** | K-composition (earlier head writes key for later head) | K-composition (L0H3 shifts residual → downstream key changes) |
| **功能** | 促进 in-context copying | 抑制 pathological repetition |
| **信号** | Positive (增强正确 token logit) | Negative (降低 repeat token logit) |

### 2.7 LoopBench 基准

| Model | Dataset | Baseline Rep Rate | Numerical Loop | Statement Loop | Hit Max Rate |
|-------|---------|------------------|----------------|----------------|-------------|
| 1.7B | LoopBench v3 (700) | 22.71% | 22.71% | 1.00% | 72.57% |
| 4B | LoopBench v3 (700) | 21.43% | 21.14% | 0.57% | 75.00% |

---

## 3. Literature Landscape (Updated)

| 论文 | 发表 | 核心发现 | 与本项目关系 |
|------|------|---------|------------|
| Copy Suppression (McDougall et al.) | ICLR 2024 | GPT-2 L10H7 通过 OV circuit 直接 suppress copy token logit | **方法论参考但机制不适用**：L0H3 不是 copy suppression head |
| Repetition Neurons (Hiraoka & Inui) | NAACL 2025 | MLP 中发现"复读神经元"，去激活可抑制复读 | **互补视角**：RSH 可能通过影响 MLP 的输入来间接工作 |
| Repeat Curse from Feature Perspective | ACL 2025 | SAE 提取"重复特征"，去激活可缓解复读 | SAE 路径可作为后续互补方法 |
| **In-context Learning and Induction Heads (Olsson et al.)** | **Transformer Circuits 2022** | **定义 induction head 的两个评分函数（prefix-matching + copying score），追踪二步 circuit（previous-token head → induction head via K-composition），训练过程中 phase change** | **方法论核心参考**：我们需要为 RSH 定义类似的 per-head mechanistic scoring functions（见 §2.5）；其 circuit tracing 方法（composition score, direct logit attribution）直接适用于 L0H3 下游路径分析 |
| Repetitions Are Not All Alike (Mahaut & Franzon) | arXiv 2025 | ICL-induced repetition 有专门 circuit；natural repetition 无明确 circuit | CoT loop 可能是第三类 repetition |
| Attention Heads Survey (Gao et al.) | Patterns 2025 | 全面 head 分类体系 | 分类框架参考 |
| When Attention Sink Emerges | ICLR 2025 | Attention sink 机制：value drain | L0H3 sink-like 行为需区分 |

---

## 4. 项目独特定位与 Paper 主叙事

### 4.1 独特定位

本项目处于尚无人系统研究的交叉地带：

- **模型**：Qwen3 系列（非 GPT-2/Llama），有 `<think>` reasoning mode
- **现象**：CoT 推理中的病态复读（think-token loop collapse），不同于一般文本生成中的重复
- **粒度**：attention head 级别的因果分析（补充 MLP neuron 和 SAE feature 的视角）
- **规模追踪**：系统性跨 0.6B–10B 追踪复读抑制机制的演变

### 4.2 建议 Paper 主叙事（mechanistic scaling story + circuit identification）

Paper 由两条交织的线索构成：

**线索 A — Circuit Identification（类比 Induction Heads 的 circuit story）**

1. **评分函数定位**：为 RSH 定义 per-head mechanistic scoring functions（DLA, SAC, DCS），在不做 ablation 的情况下提供 dense mechanistic landscape，校准后跨模型可比
2. **因果验证**：ablation 确认 L0H3 的因果必要性（1.7B rep 0%→99%），与评分函数结果交叉验证
3. **Circuit 追踪**：通过 per-component logit attribution, norm/direction decomposition 和 path patching 确定 L0H3 的具体中介路径（self-attend → output bias → downstream consumers），目标是产出可视化 circuit diagram
4. **不是 copy suppression**：明确与 McDougall et al. 的 OV-circuit direct suppression 区分，提出 residual competition 的替代机制

**线索 B — Mechanistic Scaling（跨规模演变）**

1. **在两个小模型（0.6B/1.7B）中都观察到类似的 Layer-0 suppressor-like head**（均为 L0H3），它几乎只看当前 token 的 residual stream（98.6% self-attention），但其 ablation 导致解码崩溃为复读 loop。*注意：这是功能类比而非 weight-space correspondence。*
2. **到 4B，当前方法下的单头可定位性下降**。一种可能的解释是复读抑制机制随规模从 focal 走向 distributed，但也可能反映发现方法的盲区。
3. **评分函数 landscape 的跨模型对比**：即使 4B+ 没有 clean single-head RSH，DLA landscape 仍可揭示 repeat-suppression 功能在 head 间的分布模式变化。

**优势**：
- 即使 7B/8B 没找到 clean single-head RSH，线索 B 仍然成立
- 线索 A 借鉴 Induction Heads 的方法论框架追求 circuit-level 的机制解释，独立于 scaling 结果也有价值
- 两条线索互相增强：circuit 理解使 scaling 比较更深入（不只是"有/没有 RSH"，而是"circuit 的哪个组件变了"）

### 4.3 与已有工作的结构化差异

| 维度 | McDougall (Copy Suppression) | Hiraoka (Repetition Neurons) | 本项目 |
|------|---------------------------|---------------------------|--------|
| **模型** | GPT-2 Small | GPT-2 / OPT | Qwen3 (0.6B–10B) |
| **粒度** | 单头 OV circuit | MLP neuron | Attention head + 跨规模演变 |
| **现象** | 一般文本的 token-level copy | 一般文本的重复 | CoT 推理中的 think-token loop collapse |
| **机制** | Direct logit suppression via OV | 促进复读的 neuron 激活 | **非 direct suppression**；间接 residual competition（待确认） |
| **规模** | 单模型 case study | 单模型 | 跨 0.6B–10B 的 focal→distributed transition |
| **发现** | 抑制复读的头 | 促进复读的神经元 | 抑制机制的 **scaling behavior** |

---

## 5. 核心研究问题（更新版）

### Q1: 跨模型 RSH 定位

> 在 Qwen3 系列 ≤10B 的不同规模模型中，是否普遍存在 RSH？如何可靠地定位它们？

子问题：
- 4B 为什么没有 clean RSH？是机制分布式化了，还是发现方法不适用？
- 7B/8B 的 RSH 在哪？是否仍在 Layer 0？
- RSH 的位置是否随模型规模系统性变化？

### Q2: RSH 作用机制

> L0H3 "通过改变 token 竞争格局间接影响复读"的具体路径是什么？

子问题：
- L0H3 output vector 在 residual stream 中的几何效应是什么？（norm 维持 vs 方向偏移）
- L0H3 如何与下游 layer（尤其是 MLP）交互？
- 98.6% self-attention 为什么足以抑制复读？（position-dependent bias hypothesis）

### Q3: RSH 的特异性与泛化性

> RSH 效果是否特异于"病态复读"？是否在正常重复文本中也被激活？

---

## 6. 评判体系

### 6.1 八层证据框架（含机制评分层）

| 层级 | 名称 | 标准 | 状态 |
|------|------|------|------|
| **L0** | 机制签名 | per-head mechanistic scoring（DLA, SAC, DCS）在不做 intervention 的情况下标识出该 head（类比 Induction Heads 的 prefix-matching + copying score） | **未测**（Exp 0 将填充） |
| **L1** | 统计相关性 | 复读发生时该 head 的激活模式显著区别于非复读时 | 1.7B L0H3 ✅ |
| **L2** | 因果必要性 | zero-ablation 后复读率显著上升 (p<0.05, effect size > 0.1) | 1.7B L0H3 ✅, 0.6B L0H3 ✅, 4B ❌ |
| **L3** | Circuit 刻画 | 确定完整 circuit（DLA 分解 + composition analysis + path patching），精确到可画 circuit diagram | **未测**（Exp 3B + Exp 0 DCS 将填充） |
| **L4** | 剂量效应 | scale 增加时复读率单调下降 | 1.7B L0H3 ⚠️ (benchmark-dependent) |
| **L5** | 特异性 | 对病态复读有效，对正常重复无影响 | **未测** |
| **L6** | 解码稳健性 | 效果在 greedy / sampling (temp=0.6) / top-k / top-p 等不同解码配置下稳定复现 | **未测** |
| **L7** | 跨条件稳健性 | ≥3 prompt sets, ≥2 seeds, ≥2 model sizes | 部分 (2 models pass) |

### 6.2 RSH 操作性定义

> **抑制复读头（RSH）** 是指满足以下条件的 attention head：
> 当模型处于"即将进入病态复读"的临界状态时，该 head 的正常运行是阻止模型陷入复读循环的因果必要条件之一。

与相邻概念的区分：

| 概念 | L0H3 证据 | 结论 | 排除强度 |
|------|----------|------|---------|
| Copy Suppression Head | 静态 OV top-10 suppression=0% | **初步排除**（不排除 context-conditional / LN-amplified 的间接 logit 效应） | 中 |
| Self-correction / Wait Head | wait logit delta=-3.46 | 相关但非同一概念 | — |
| Length-shortening Head | scale 后长度缩短 | 副作用，非机制 | — |
| Induction Head | 98.6% self-attention | **排除** | 强 |
| Attention Sink (value drain) | 98.6% self-attention, sink-like | **未区分**：L0H3 的抑制效果可能是 value drain 的副作用 | 弱 |
