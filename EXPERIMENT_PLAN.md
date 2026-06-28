# Experiment Plan

**项目**: 基于 LoopBench-inspired 数据系统定位并验证“抑制复读头”  
**日期**: 2026-04-18  
**主模型**: `Qwen/Qwen3-1.7B`  
**状态**: Draft for review  
**主数据源**: [evaluation/data/loopbench_inspired/test.jsonl](/home/suyc24/Python/Qwen2.5-Math/evaluation/data/loopbench_inspired/test.jsonl)  

## 1. 研究目标

本轮实验不再把“会缩短输出”或“会减少 wait/reflection”的 head 直接当成目标，而是要回答两个更严格的问题：

1. `Qwen3-1.7B` 中是否存在一个以上、对病态复读有稳定因果作用的 attention head？
2. 如果某个 head 被称为“抑制复读头”，它是否满足：
   - 对病态复读而非一般短输出有效
   - 对 loop-vs-escape token 竞争有可测的因果影响
   - 对正常 copy / boundary / hesitation 现象不只是同一个机制的别名

本计划的核心策略是：

- 先用 `LoopBench-inspired` 数据挖出新的复读种子池
- 再在 `1.7B` 已知 `29` 个 local heads 上做系统性因果扫描
- 最后用 logit-margin、specificity probe、clean benchmark side-effect 三层证据筛选真正候选

## 2. Claim Map

| Claim | 为什么重要 | 最低可接受证据 | 对应实验 |
| --- | --- | --- | --- |
| C1 | `LoopBench-inspired` 能稳定产出可复用的 loop 种子，而不是只是一套长推理题 | 在 `700` 题上挖出足够数量的 raw loop；经 HF greedy 重跑后仍保留 `>=30` 个 stable loop case | E0, E1 |
| C2 | `1.7B` 中不止 `L0H3` 一种 head 值得进入 anti-repetition 候选池，或者至少可以系统证明“只有 L0H3 强” | 在 `29` 个 local heads 中做统一 scan，得到可排序 shortlist，而不是只讨论 `L0H3` | E2 |
| C3 | 强候选 head 的作用不是“单纯缩短输出”，而是改变 loop/escape 竞争 | 首分叉点 `escape_minus_loop_logit` 在 scale 后稳定上升，且在中等 scale 就跨过 0 | E3 |
| C4 | 强候选 head 不是普通 copy-suppression / boundary / hesitation head | 在 boundary probe 和 clean benchmark 上，效应模式与 loop rescue 明显不同 | E4, E5 |
| C5 | 结论不只对 LoopBench 一套 prompt 成立 | 在现有 NuminaMath repetition pool 上方向一致 | E6 |

## 3. 已有结果对本计划的约束

以下结论已经比较清楚，计划应避免重复验证：

- `L0H3` 是 `1.7B` 上最强的 wait-suppression head，这一点已经足够明确。
- `L0H3` 在 `</think>` 位置是典型 `self_local`，不是普通 `recent_local` copy head。
- `copy suppression` 和 `OV circuit` 目前都不支持“L0H3 是直接 copy suppressor”。
- `L0H3` 对 repetition 的最佳证据来自首分叉点的 token 竞争变化，而不是一般性的长度缩短。
- `4B` 目前没有清晰复现出 `1.7B / L0H3` 这种强机制头；继续大规模扫 `4B` 不是当前最高优先级。

因此，本计划的主线是：

- **先做 `1.7B + LoopBench + local head pool`**
- **先建稳定种子池和判据**
- **先把“特异性”和“因果性”补齐**
- **暂不把 4B 当 must-run**

## 4. 操作性定义与统一判据

### 4.1 病态复读 case

所有复读判断必须走 `cot_research.repetition_analysis`，不允许新写 ad hoc heuristics。

本计划中的 case 分层：

- `raw_loop_case`:
  - 在一次 baseline 生成上被 `scripts/filter_repetition_cot.py` 判为复读
- `stable_loop_case`:
  - 对同一题在 HF greedy 条件下重跑一次，仍判为复读
- `clean_control_case`:
  - baseline 不复读，且与 stable loop set 在子任务分布和长度分布上尽量匹配

### 4.2 “抑制复读头”操作性定义

一个 head 只有在同时满足下列条件时，才进入强候选：

- `scale-up` 能降低 `stable_loop_case` 的复读率，或提升首分叉点 `escape_minus_loop_logit`
- `zero` 或 `scale-down` 至少在一部分 case 上表现出反方向作用
- 对 clean controls 不主要表现为粗暴提前结束
- 在 boundary/copy probe 上不更像 generic copy head

### 4.3 本轮不直接声称的内容

- 不直接声称“模型里普遍存在一类 anti-repetition heads”
- 不直接声称“L0H3 已被完全机制化解释”
- 不直接用 `LoopBench-inspired` 做准确率主 benchmark

原因：

- 当前 [evaluation/data/loopbench_inspired/test.jsonl](/home/suyc24/Python/Qwen2.5-Math/evaluation/data/loopbench_inspired/test.jsonl) 的顶层 `correct_answer` 仍为空，主用途是 loop mining，不是 accuracy leaderboard

## 5. 共享配置

| 项目 | 默认配置 | 备注 |
| --- | --- | --- |
| 主模型 | `Qwen/Qwen3-1.7B` | 主发现阶段只做 1.7B |
| bulk generation backend | `vLLM` | 只用于高吞吐 baseline CoT 采集 |
| intervention/logit backend | `HF` | 所有 head 干预、logit capture 统一用 HF |
| decoding | greedy | `--no-do_sample`；保持条件一致 |
| 主 system prompt | `LoopBench-style meticulous reasoning` | 建议从 row `metadata.recommended_system_prompt` 读取 |
| assistant prefix | `<think>\n` | 与仓库现有脚本一致 |
| screening `max_new_tokens` | `4096` | LoopBench 挖种子主配置 |
| confirmation `max_new_tokens` | `4096` 或 `8192` | 只在种子不足时上调 |
| repetition thresholds | `40/6/8/24`, line `4`, word `5/8/24` | 与现有 repetition pipeline 对齐 |
| scale shortlist | `1.2, 1.5, 2.0` | `4.0` 只放到机制 sweep |
| 随机种子 | `1234` | greedy 条件下主要用于抽样一致性 |

### 5.1 1.7B local head 池

来自：
[experiment_results/experiments/phase5_head_locality/head_locality_classification_qwen3_1p7b_20260414_1/data/classification/local_heads.csv](/home/suyc24/Python/Qwen2.5-Math/experiment_results/experiments/phase5_head_locality/head_locality_classification_qwen3_1p7b_20260414_1/data/classification/local_heads.csv)

共 `29` 个：

`L0H9, L0H7, L0H5, L0H15, L0H14, L0H1, L0H3, L0H12, L2H9, L1H14, L1H3, L1H2, L1H6, L4H8, L0H2, L2H12, L1H7, L3H8, L2H4, L1H1, L2H1, L0H8, L24H0, L4H4, L4H1, L5H13, L0H10, L2H13, L4H11`

高优先级子集：

- `L0H3`
- `L1H3`
- `L1H2`
- `L0H2`
- `L2H12`
- `L2H1`
- `L2H13`
- `L4H1`
- `L0H9`
- `L0H7`
- `L0H5`
- `L0H15`

理由：

- 前 8 个更像 `recent_local` 或已知机制相关局部头
- 后 4 个是最强 `self_local` 短输出头，用于排除“只是 generic shortener”

## 6. 执行前需要的薄封装

这些不是研究假设本身，但会直接影响实验能否按仓库规范落地。

### P0. 通用 JSONL CoT 采集脚本

**状态**: 需要新增 thin script  
**建议文件**: `scripts/collect_jsonl_cot.py`

建议直接从 [scripts/collect_qwen3_numinamath_cot.py](/home/suyc24/Python/Qwen2.5-Math/scripts/collect_qwen3_numinamath_cot.py) 轻量泛化，而不是重写一套。

最低要求：

- 输入任意含 `id/question/metadata` 的 JSONL
- 支持 `--system_prompt_mode fixed|metadata`
- 输出字段兼容 `filter_repetition_cot.py`
- 支持 `--resume`
- backend 仍用 `vLLM`

### P1. stable repetition seed pool builder

**状态**: 建议新增 thin script  
**建议文件**: `scripts/build_stable_repetition_seed_pool.py`

最低要求：

- 输入 raw repetition cases JSONL
- 对同题做一次 HF greedy baseline rerun
- 输出：
  - `stable_loop_cases.jsonl`
  - `clean_controls.jsonl`
  - `summary.json`

### P2. LoopBench ground-truth 提升

**状态**: 非阻塞，但建议后续补  

原因：

- 现在 `LoopBench-inspired` 主要适合 loop mining
- 如果后面想在 LoopBench 上报 accuracy，需要把 solver answer 提升到统一 judge 接口，而不是继续留在 `metadata`

## 7. Experiment Blocks

## E0. LoopBench baseline collection

**目的**: 在 `LoopBench-inspired` 上收集 baseline CoT，确认它是否真的能给出新的 loop seed  
**优先级**: MUST-RUN  
**可执行性**: 需要先完成 `P0`

### 输入

- full set:
  - [evaluation/data/loopbench_inspired/test.jsonl](/home/suyc24/Python/Qwen2.5-Math/evaluation/data/loopbench_inspired/test.jsonl)
- smoke:
  - [evaluation/data/loopbench_inspired/smoke_questions.jsonl](/home/suyc24/Python/Qwen2.5-Math/evaluation/data/loopbench_inspired/smoke_questions.jsonl)

### 脚本

- `[NEW]` `scripts/collect_jsonl_cot.py`
- existing `scripts/filter_repetition_cot.py`

### Pilot 配置

- examples: `14`
- backend: `vLLM`
- decoding: greedy
- `max_new_tokens=1024`
- `system_prompt_mode=metadata`
- 输出:
  - `outputs/loopbench_cot/qwen3_1p7b_loopbench_smoke_20260418_1/`

### Full 配置

- examples: `700`
- backend: `vLLM`
- decoding: greedy
- `max_new_tokens=4096`
- `request_batch_size=128` 或 `256`
- `system_prompt_mode=metadata`
- 输出:
  - `outputs/loopbench_cot/qwen3_1p7b_loopbench_full_20260418_1/`

### 主要指标

- raw repetition count
- repetition rate by subtask
- `generated_tokens` 分布
- `hit_max_new_tokens` rate

### 决策门槛

- go:
  - raw repetition cases `>=15`
- conditional go:
  - raw repetition cases `<15`，但明显集中在某几个子任务，可继续做 targeted rerun
- stop / redesign:
  - raw repetition cases 很少且无明显集中；此时不应直接进入 head scan

### 失败时的 fallback

- 只对 `square_root / long_division / newtons_iteration / tower_of_hanoi / path_planning` 再跑一轮 `max_new_tokens=8192`
- 若仍不足，则先把 LoopBench 作为辅助集，主 discovery 继续依赖现有 NuminaMath stable pool

## E1. Stable loop / clean control pool construction

**目的**: 从 `raw_loop_case` 变成后续可复用的、head-neutral 的 stable pool  
**优先级**: MUST-RUN  
**可执行性**: 需要先完成 `P1`

### 输入

- `E0` 产出的 full baseline generation JSONL
- `E0` 过滤出的 raw repetition cases

### 脚本

- `[NEW]` `scripts/build_stable_repetition_seed_pool.py`
- 内部复用 `cot_research.repetition_analysis`

### Pilot 配置

- raw loop subset: `20`
- controls: `20`
- backend: `HF`
- decoding: greedy
- `max_new_tokens=2048`

### Full 配置

- raw loop cases: `all`
- control selection:
  - 与 stable loops 在 `subtask` 分布尽量匹配
  - 与 baseline `generated_tokens` 分位点尽量匹配
- backend: `HF`
- decoding: greedy
- `max_new_tokens=4096`

### 输出

- `outputs/loopbench_seed_sets/qwen3_1p7b_20260418_1/stable_loop_cases.jsonl`
- `outputs/loopbench_seed_sets/qwen3_1p7b_20260418_1/clean_controls.jsonl`
- `outputs/loopbench_seed_sets/qwen3_1p7b_20260418_1/summary.json`

### 主要指标

- stable loop count
- stable retention rate
- clean control count
- stable loop 的 subtask 分布

### 决策门槛

- go:
  - stable loop cases `>=30`
- conditional go:
  - stable loops `15-29`
  - 此时 discovery 仍可做 pilot，但 full sweep 暂缓
- stop:
  - stable loops `<15`

## E2. 1.7B local-head scale rescue sweep

**目的**: 在统一 head pool 上系统定位候选，而不是只看 `L0H3`  
**优先级**: MUST-RUN  
**可执行性**: 现有脚本可直接跑

### 输入

- `E1` 的 `stable_loop_cases.jsonl`
- `E1` 的 `clean_controls.jsonl`

### 脚本

- [scripts/test_head_boost_effects.py](/home/suyc24/Python/Qwen2.5-Math/scripts/test_head_boost_effects.py)

### Pilot 配置

- head set: 高优先级 `12` 个
- eval rows:
  - stable loops `24`
  - clean controls `24`
- intervention: `scale`
- scale: `1.2`
- `baseline_mode=rerun`
- `max_new_tokens=2048`
- repetition thresholds:
  - `same_token_run_threshold=40`
  - `tail_repeat_min_repeats=6`
  - `tail_repeat_max_ngram=8`
  - `tail_repeat_min_span=24`
  - `line_repeat_threshold=4`
  - `word_tail_repeat_min_repeats=5`
  - `word_tail_repeat_max_ngram=8`
  - `word_tail_repeat_min_span=24`

### Full 配置

- head set: 全部 `29` 个 local heads
- eval rows:
  - stable loops `40`
  - clean controls `40`
- intervention: `scale`
- scale: `1.2`
- `baseline_mode=rerun`
- `max_new_tokens=4096`

### 排名指标

- `loop_rescue_rate`
  - baseline loop, intervention non-loop
- `control_induced_rate`
  - baseline clean, intervention loop
- `loop_token_delta`
  - stable loops 上的 token 数变化
- `control_token_delta`
  - clean controls 上的 token 数变化

### 候选排序规则

按以下优先级排序：

1. `loop_rescue_rate` 高
2. `control_induced_rate` 低
3. `control_token_delta` 不应表现为极端提前结束
4. 若并列，优先保留 `recent_local` 头，再保留 `self_local` 头

### 决策门槛

- shortlist:
  - `loop_rescue_rate >= 0.10`
  - `control_induced_rate <= 0.05`
- 强 shortlist:
  - `loop_rescue_rate >= 0.15`
  - `control_induced_rate <= 0.03`

### 预期结果解释

- 如果只有 `L0H3` 达标，这是一个有效结果，不是失败
- 如果多个 recent-local heads 达标，说明“anti-repetition”很可能不是单一 self-local 机制
- 如果只有 self-local heads 达标，后续必须重点排除“generic shortener”

## E3. Shortlist zero sweep and opposite-direction check

**目的**: 给出必要性证据，至少证明 zero 与 scale 的方向不一致  
**优先级**: MUST-RUN  
**可执行性**: 现有脚本可直接跑

### 输入

- `E2` shortlist heads
- 与 `E2` 相同的固定 eval rows

### 脚本

- [scripts/test_head_boost_effects.py](/home/suyc24/Python/Qwen2.5-Math/scripts/test_head_boost_effects.py)

### 配置

- heads: `E2` top `5`
- intervention: `zero`
- `baseline_mode=rerun`
- eval rows:
  - stable loops `40`
  - clean controls `40`
- `max_new_tokens=4096`
- 其余 repetition thresholds 与 `E2` 相同

### 主要指标

- zero 后 `control_induced_rate`
- zero 后 stable loops 的长度和 repetition severity 是否进一步恶化
- 与 `E2 scale=1.2` 的方向是否相反

### 通过标准

- 至少一部分 head 在 zero 与 scale 上呈现反方向变化
- 如果一个 head 只有 scale 有效、zero 完全无效，仍可保留为“部分充分但必要性弱”

## E4. First-divergence logit-margin sweep

**目的**: 判断候选是否真的改变 loop-vs-escape 竞争  
**优先级**: MUST-RUN  
**可执行性**: 现有脚本可直接跑

### 输入

- `E1` 的 `stable_loop_cases.jsonl`

### 脚本

- [scripts/run_repetition_scale_logit_sweep.py](/home/suyc24/Python/Qwen2.5-Math/scripts/run_repetition_scale_logit_sweep.py)

### Pilot 配置

- heads: `E2` top `5`
- `max_cases=30`
- `scale_values=1.2,1.5,2.0`
- `top_k=10`
- `max_new_tokens=4096`
- `attn_implementation=eager`

### Full 配置

- heads: `E2/E3` 后剩余 top `2`
- `max_cases=all`，上限建议 `100`
- `scale_values=1.2,1.5,2.0,4.0`
- `top_k=10`
- `max_new_tokens=4096`

### 主要指标

- `escape_minus_loop_logit`
- `loop_logit_delta_vs_baseline`
- `escape_logit_delta_vs_baseline`
- `both_effect_rate`
- repetition rate by scale

### 强候选标准

- `escape_minus_loop_logit` 在 `1.2` 或 `1.5` 处跨过 `0`
- 平均 margin 随 scale 增大而增大
- `both_effect_rate` 在中等 scale 不低于 `0.4`

### 明确排除的情形

- 只看到 `generated_tokens` 下降，但 `escape_minus_loop_logit` 不改善
- 只在 `4.0` 这种明显高 scale 才出现不稳定改善

## E5. Specificity probes

**目的**: 排除“只是 copy/boundary/early-close 头”  
**优先级**: MUST-RUN  
**可执行性**: 现有脚本可直接跑

### E5-A. Boundary / copy specificity

**脚本**: [scripts/run_boundary_head_probe.py](/home/suyc24/Python/Qwen2.5-Math/scripts/run_boundary_head_probe.py)

**输入**:

- [evaluation/data/self_correction_ablation/test_questions.jsonl](/home/suyc24/Python/Qwen2.5-Math/evaluation/data/self_correction_ablation/test_questions.jsonl)

**配置**:

- heads: top `2` candidates + `L0H3` reference
- `scale_values=1.2,1.5`
- `sharp_phrase_count=96`
- `wrong_tail_count=96`
- `control_count=96`
- `include_direct_write`

**关键指标**:

- `target_prob_delta`
- `target_control_logit_gap_delta`
- `eos_logit_delta`
- direct write stats

**解释规则**:

- 如果某个 head 在 boundary probe 上极强、但 loop rescue 弱，更像 copy/boundary head
- 如果 loop rescue 强、但 boundary probe 只是中等或有限，才更接近 anti-repetition head

### E5-B. Clean benchmark side-effect scan

**脚本**: [scripts/run_l0h3_scale_wait_length.py](/home/suyc24/Python/Qwen2.5-Math/scripts/run_l0h3_scale_wait_length.py)

**输入**:

- [evaluation/data/self_correction_ablation/test_questions.jsonl](/home/suyc24/Python/Qwen2.5-Math/evaluation/data/self_correction_ablation/test_questions.jsonl)

**配置**:

- heads: top `5`
- scales: `1.0,1.2`
- `max_examples=200`
- `max_new_tokens=4096`
- greedy

**关键指标**:

- correctness delta
- generated token delta
- reflection delta
- clean-set repetition delta

**通过标准**:

- accuracy drop 不明显
- clean-set induced repetition 低
- 如果 head 只是让所有题都短很多，且正确率下降明显，则判为 generic shortener

## E6. Auxiliary cross-dataset robustness

**目的**: 检查候选是否只对 LoopBench 特异，还是也能作用于已有 repetition pool  
**优先级**: SHOULD-RUN  
**可执行性**: 现有脚本可直接跑

### 输入

- 现有 repetition seed pool:
  - `outputs/repetition/all_repetition_cases.jsonl`

### 脚本

- [scripts/run_repetition_scale_logit_sweep.py](/home/suyc24/Python/Qwen2.5-Math/scripts/run_repetition_scale_logit_sweep.py)
- [scripts/test_head_boost_effects.py](/home/suyc24/Python/Qwen2.5-Math/scripts/test_head_boost_effects.py)

### 配置

- heads: LoopBench 阶段最终 top `2`
- scale values: `1.2,1.5,2.0`
- `max_cases=203` 作为上限目标；实际以脚本重筛 stable cases 为准
- greedy

### 成功标准

- 方向一致即可，不要求 effect size 与 LoopBench 完全一致
- 若方向相反，优先认为该 head 是 dataset-specific，而非普适 anti-repetition head

## E7. Optional cross-model transfer

**目的**: 在 `0.6B / 4B` 上只做极小规模验证，不再大扫  
**优先级**: NICE-TO-HAVE  
**执行前提**:

- `1.7B` 至少出现 `>=1` 个强候选
- LoopBench 与 NuminaMath 两边方向一致

### 候选模型 / 头

- `Qwen/Qwen3-0.6B`:
  - 优先 `L0H3`
- `Qwen/Qwen3-4B`:
  - 只看已有 4B local head scan 中最强几个：
    - `L15H10`
    - `L2H25`
    - `L1H29`
    - `L0H18`

### 原则

- 这一步只做“弱复现检查”
- 不在本轮把 4B 重新当主战场

## 8. Run Order and Milestones

| Milestone | 目标 | 主要实验 | Stop / Go Gate | 预算判断 |
| --- | --- | --- | --- | --- |
| M0 | 代码 readiness | `P0`, `P1` smoke | 新脚本能 `py_compile`、`--help`、smoke 通过 | 低 |
| M1 | LoopBench 挖种子 | `E0`, `E1` | stable loops `>=30` | 低到中 |
| M2 | 候选发现 | `E2` | 至少得到一个明确 shortlist；如果只有 `L0H3` 也接受 | 中 |
| M3 | 必要性与机制 | `E3`, `E4` | top `2` 中至少一个 head 有清晰 margin evidence | 中到高 |
| M4 | 特异性过滤 | `E5` | 排除 generic shortener / copy head 假阳性 | 中 |
| M5 | 稳健性 | `E6` | LoopBench 与 NuminaMath 方向一致 | 中 |
| M6 | 可选迁移 | `E7` | 只在前面结果很干净时做 | 可选 |

## 9. Remote Execution Policy 对应执行方式

所有 full runs 必须遵守 [AGENT_GUIDE.md](/home/suyc24/Python/Qwen2.5-Math/AGENT_GUIDE.md) 第 7 节：

1. `py_compile`
2. `--help`
3. smoke run
4. `tmux` full run
5. 日志落到 `logs/`
6. 确认 GPU / 输出文件真的在更新

### 9.1 统一远程前置命令

```bash
cd /home/yucheng/experiment/Qwen2.5-Math
eval "$(/home/yucheng/bin/micromamba shell hook -s bash)"
micromamba activate qwen_math
```

### 9.2 现有脚本的 smoke/full 模板

#### `filter_repetition_cot.py`

```bash
python -m py_compile scripts/filter_repetition_cot.py
python scripts/filter_repetition_cot.py --help
python scripts/filter_repetition_cot.py \
  --input_jsonl outputs/loopbench_cot/qwen3_1p7b_loopbench_smoke_20260418_1/rows.jsonl \
  --output_jsonl outputs/repetition/loopbench_smoke_20260418_1/repetition_cases.jsonl \
  --summary_json outputs/repetition/loopbench_smoke_20260418_1/repetition_summary.json \
  --top_markdown outputs/repetition/loopbench_smoke_20260418_1/repetition_top.md
```

#### `test_head_boost_effects.py`

```bash
python -m py_compile scripts/test_head_boost_effects.py
python scripts/test_head_boost_effects.py --help
python scripts/test_head_boost_effects.py \
  --input_source jsonl \
  --input_jsonl outputs/loopbench_seed_sets/qwen3_1p7b_20260418_1/eval_48.jsonl \
  --output_dir outputs/head_boost_effects/loopbench_qwen3_1p7b_l0h3_scale1p2_smoke \
  --head_labels L0H3 \
  --intervention_kind scale \
  --scale 1.2 \
  --baseline_mode rerun \
  --max_examples 48 \
  --max_new_tokens 2048 \
  --parallel_gpu_ids <GPU_IDS> \
  --parallel_workers <N> \
  --same_token_run_threshold 40 \
  --tail_repeat_min_repeats 6 \
  --tail_repeat_max_ngram 8 \
  --tail_repeat_min_span 24 \
  --line_repeat_threshold 4 \
  --word_tail_repeat_min_repeats 5 \
  --word_tail_repeat_max_ngram 8 \
  --word_tail_repeat_min_span 24 \
  --min_trigger_count 1 \
  --no-do_sample
```

#### `run_repetition_scale_logit_sweep.py`

```bash
python -m py_compile scripts/run_repetition_scale_logit_sweep.py
python scripts/run_repetition_scale_logit_sweep.py --help
python scripts/run_repetition_scale_logit_sweep.py \
  --repeat_examples_jsonl outputs/loopbench_seed_sets/qwen3_1p7b_20260418_1/stable_loop_cases.jsonl \
  --output_dir outputs/repetition_scale_logit/loopbench_qwen3_1p7b_l0h3_smoke \
  --head_label L0H3 \
  --scale_values 1.2,1.5,2.0 \
  --max_cases 10 \
  --max_new_tokens 4096 \
  --parallel_gpu_ids <GPU_IDS> \
  --parallel_workers <N>
```

#### `run_boundary_head_probe.py`

```bash
python -m py_compile scripts/run_boundary_head_probe.py
python scripts/run_boundary_head_probe.py --help
python scripts/run_boundary_head_probe.py \
  --cot_input_jsonl evaluation/data/self_correction_ablation/test_questions.jsonl \
  --output_dir outputs/boundary_probe/qwen3_1p7b_l0h3_smoke \
  --model_name_or_path Qwen/Qwen3-1.7B \
  --head_label L0H3 \
  --scale_values 1.2,1.5 \
  --sharp_phrase_count 8 \
  --wrong_tail_count 8 \
  --control_count 8 \
  --parallel_gpu_ids <GPU_IDS> \
  --parallel_workers <N>
```

## 10. 当前不建议投入的方向

- 继续用 `wait` 排名直接当 anti-repetition 发现器
- 继续先做 `4B` 大规模全头扫描
- 继续优先做 `OV / copy suppression` 机制验证，而不先建稳定 LoopBench seed pool
- 在 `LoopBench-inspired` 还没补统一 `correct_answer` 时，把它当 accuracy 主 benchmark
- 把 `scale=4.0` 以上的高倍率行为当作主证据

## 11. 本轮计划的关键风险

### R1. LoopBench 可能没有预期中那么容易诱发 loops

**缓解**:

- 先做 `E0/E1`，不直接跳进 head scan
- 若 stable loops 不足，LoopBench 先降级为辅助集

### R2. 候选头可能全部退化成“generic shortener”

**缓解**:

- 必须跑 `E5-B`
- 用 clean-set induced repetition 和 accuracy side-effect 过滤

### R3. `E2` 计算量过大

**缓解**:

- 先 `12 heads x 48 rows` pilot
- 再扩到 `29 heads x 80 rows`

### R4. zero 证据可能偏弱

**缓解**:

- 不把 zero 单独当唯一必要性证据
- 与 `E4` 的 first-divergence margin 一起看

## 12. 我建议你重点审核的 5 个点

1. 是否接受把 `LoopBench-inspired` 的当前角色定为“loop mining 主数据”，而不是 accuracy 主 benchmark  
2. `E2` 的 head pool 是否只扫 `29` 个 local heads，还是要扩到更大范围  
3. `E2` full 配置是 `40+40` 还是再压缩到 `30+30`  
4. `E4` 是否保留 `scale=4.0` 作为高倍率诊断，而不是主证据  
5. 是否同意把 `4B/0.6B` 全部后移到 `E7`

## 13. 审核后建议的第一批远程实验

如果这份计划通过，我建议第一批只启动下面 3 个：

1. `P0 + E0 smoke/full`
2. `P1 + E1 full`
3. `E2 pilot` 只扫高优先级 `12` 个 heads

原因：

- 这三步能最快回答“LoopBench 是否真的能提供有效 seed pool”
- 也能最快回答“除了 `L0H3` 之外，是否立刻出现第二个像样候选”
- 在这三步之前，不值得开更贵的 logit sweep 或 boundary probe
