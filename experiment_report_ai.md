# Self-Correction Ablation 实验报告

## 1. 报告范围与数据完整性

本报告分析四组实验输出：

- `outputs_1.7b`
- `outputs_1.7b_L0H3`
- `outputs_4b`
- `outputs_4b_L0H22`

其中 outputs_1.7b 和 outputs_4b 是针对所有head在12道题目上实验的，outputs_1.7b_L0H3和outputs_4b_L0H22是针对屏蔽特定head在在GSM8K全集上跑的

## 2. 核心结论

### 2.1 最稳定的发现是 `Qwen3-1.7B / L0H3`

- 在 `12` 题全 head 扫描中，`L0H3` 已经是最强破坏 head
- 在 `1319` 题 GSM8K 风格数据上，屏蔽 `L0H3` 后 corrected rate 从 `0.877` 暴跌到 `0.072`
- 失败模式不是简单地“保留错误答案”，而是大规模退化为 `no_boxed` 或生成重复字符

### 2.2 `Qwen3-4B / L0H22` 不是 4B 上的全局关键 head

`L0H22` 在某些单题上非常强，但从全量 1319 题看，它不是 4B 的“1.7B-L0H3 对应物”

- 在 `outputs_4b_L0H22` 中，平均 wait-logit 变化只有 `-0.382`
- 反推 outcome 后，`1200/1319` 题仍然 corrected，比例约 `0.910`
- 这说明它更像是一个“有贡献但非瓶颈”的支持性 head

### 2.3 4B 的自我纠错机制更像“分布式头簇”，而不是单个早层 head

`outputs_4b` 的 6 个单题扫描虽然不规范，但反复出现的高影响 head 不是只有 `L0H22`，而是：

- `L34H14`
- `L0H1`
- `L20H7`
- `L20H9`
- `L15H9`
- `L22H12`
- `L22H15`

这更像是一个跨层、偏中后层的 head 簇，而不是单点控制。

### 2.4 `wait logit` 是有用代理指标，但不是充分指标

已有数据不支持“谁让 wait logit 变化最大，谁就是最关键行为 head”。

- 在 `outputs_1.7b` 中，wait-logit 变化幅度与 corrected-rate 下降的 Pearson 相关只有 `0.353`
- 这属于“有关系，但远远不够强”
- 例如 `L24H6` 的平均 wait-logit 变化最大，但行为破坏并不如 `L0H3` 明显

因此，后续选 head 不能只看 `delta(wait_logit)`，必须结合行为指标。

在 `1.7B / L0H3` 和 `4B / L0H22` 的异常样本中，都出现了大量：

- 反复重复 `Wait`
- 长文本循环
- 不闭合 `</think>`
- 最终没有 `\\boxed{}`

## 3. 分组结果与深入分析

### 3.1 `outputs_1.7b`：12 题、全 448 个 head 扫描

配置：

- 模型：`Qwen/Qwen3-1.7B`
- 样本数：`12`
- 测试 head 数：`448`

baseline：

- corrected rate：`1.000`
- keyword-hit rate：`1.000`

按 corrected rate 排序，破坏最强的 head：

| head | corrected rate | no_boxed | other_answer | keep_wrong |
| --- | ---: | ---: | ---: | ---: |
| `L0H3` | 0.083 | 11 | 0 | 0 |
| `L0H7` | 0.583 | 1 | 4 | 0 |
| `L10H6` | 0.750 | 0 | 0 | 3 |
| `L0H2` | 0.917 | 0 | 0 | 1 |
| `L12H14` | 0.917 | 0 | 0 | 1 |

按平均行为破坏强度看，层分布很不均匀。行为破坏最强的层是：

| layer | affected heads | strong heads | mean behavior drop | mean abs wait delta |
| --- | ---: | ---: | ---: | ---: |
| `0` | 3 | 2 | 0.0885 | 0.4280 |
| `10` | 1 | 1 | 0.0156 | 0.1906 |
| `12` | 1 | 0 | 0.0052 | 0.2112 |
| `13` | 1 | 0 | 0.0052 | 0.2537 |
| `14` | 1 | 0 | 0.0052 | 0.2213 |
| `15` | 1 | 0 | 0.0052 | 0.1976 |

这里最值得注意的是：

- `Layer 0` 的平均行为破坏强度远高于其它层
- `L0H3` 与 `L0H7` 都在第 0 层

另一个值得注意的点是不同失败类型：

- `L0H3` 主要导致模型陷入重复回答
- `L10H6` 更像是导致“保留错误答案”

也就是说，不同 head 打掉的不是同一个子功能：

- 有的 head 更像影响“重新检查并收束”
- 有的 head 更像影响“最终答案是否切回正确值”

### 3.2 `outputs_1.7b_L0H3`：1319 题、只测 `L0H3`

配置：

- 模型：`Qwen/Qwen3-1.7B`
- 样本数：`1319`
- head：`L0H3`
- 并行方式：`example`
- Stage 并行 worker：`8`

整体结果：

| 条件 | corrected | keep_wrong | other_answer | no_boxed | corrected rate | keyword-hit rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 1157 | 34 | 122 | 6 | 0.877 | 0.995 |
| ablate `L0H3` | 95 | 0 | 219 | 1005 | 0.072 | 0.397 |

这是一个极强的结果，但更关键的是“怎么坏掉的”。

按 outcome 分组后的统计如下：

| outcome | count | mean delta | median delta | mean len | keyword rate | think close rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| corrected | 95 | -1.650 | -2.609 | 3813 | 0.274 | 0.789 |
| other_answer | 219 | -2.077 | -2.516 | 6134 | 0.447 | 0.457 |
| no_boxed | 1005 | -1.308 | -1.828 | 10476 | 0.390 | 0.346 |

这里有三条非常重要的结论。

第一，`L0H3` 的主失败模式是“生成结构崩坏”，不是“稳稳保留错误答案”。

- `keep_wrong = 0`
- `no_boxed = 1005`

也就是说，屏蔽 `L0H3` 之后，模型最常见的状态不是坚持错误答案，而是根本无法完成一个正常的自我纠错收束过程。

第二，`no_boxed` 样本平均长度极长，说明出现了明显的退化循环。

- corrected 平均长度约 `3813`
- no_boxed 平均长度约 `10476`

这和你在原始 CoT 中能看到的大量重复、拖长、断裂格式是一致的。`L0H3` 很可能参与了“让反思过程及时闭合”的控制。

第三，`keyword_hit` 在这里会误导判断。

直觉上，好像 corrected 样本应该更常出现 `wait`、`hold on` 这类词，但实际不是：

- corrected 的 keyword rate 只有 `0.274`
- no_boxed 的 keyword rate 反而是 `0.390`
- other_answer 的 keyword rate 是 `0.447`

这说明当输出进入退化循环时，关键词会被机械重复，导致 `keyword_hit` 仍然为真，但其实模型并没有真正完成有效反思。

因此，对 `1.7B / L0H3` 的更准确解释是：

- `L0H3` 不只是“提升 wait token”
- 它更像是在维持一个可收束的反思轨道
- 一旦屏蔽，模型容易进入反思样式的结构性退化

### 3.3 `outputs_4b_L0H22`：1319 题、只测 `L0H22`

配置：

- 模型：`Qwen/Qwen3-4B`
- 样本数：`1319`
- head：`L0H22`
- 并行方式：`example`
- Stage 并行 worker：`8`

因为缺少 `head_summary.csv`，这里的 outcome 是从 `ablation_L0H22_CoT.jsonl` 反推的。

反推结果：

| outcome | count | rate |
| --- | ---: | ---: |
| corrected | 1200 | 0.910 |
| keep_wrong | 33 | 0.025 |
| other_answer | 66 | 0.050 |
| no_boxed | 20 | 0.015 |

wait-logit 总体统计：

- mean delta：`-0.382`
- mean abs delta：`0.461`
- negative delta 样本数：`1202`
- positive delta 样本数：`112`

如果只看这几行，会得到一个普通结论：`L0H22` 有效，但不强。  
但更深一层看，关键信息在 outcome 分组里。

| outcome | count | mean delta | mean baseline wait | mean ablated wait | mean len | keyword rate | think close rate | starts with wait |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| corrected | 1200 | -0.404 | 26.853 | 26.449 | 1754 | 0.722 | 0.985 | 0.710 |
| keep_wrong | 33 | 0.585 | 24.489 | 25.074 | 2806 | 0.212 | 0.788 | 0.212 |
| other_answer | 66 | -0.414 | 27.181 | 26.767 | 3101 | 0.788 | 0.985 | 0.758 |
| no_boxed | 20 | -0.509 | 24.605 | 24.096 | 10453 | 1.000 | 0.050 | 0.950 |

这里有四条很关键的结论。

第一，`keep_wrong` 样本的平均 delta 竟然是正的。

- corrected：mean delta = `-0.404`
- keep_wrong：mean delta = `+0.585`

这意味着：在真正失败的那部分样本里，`L0H22` 并不是“被屏蔽后 wait logit 降太多，所以不纠错”。  
相反，这些失败样本更像是本来就不太进入强反思状态，或者该 head 在这些样本上并不承担主要责任。

第二，`keep_wrong` 样本本身的 baseline wait logit 就更低。

- corrected 的 baseline wait = `26.853`
- keep_wrong 的 baseline wait = `24.489`

这说明 4B 上真正决定 self-correction 成败的，可能不是 `L0H22` 单独是否工作，而是更广泛的 reflective subcircuit 是否被激活。

第三，corrected 样本的 reflective pattern 仍然非常完整。

- `71.0%` 的 corrected 样本以 `Wait...` 风格开头
- `98.5%` 的 corrected 样本能正常闭合 `</think>`

所以屏蔽 `L0H22` 后，4B 大多数情况下仍能稳定进入并完成反思过程。

第四，4B 也会出现退化循环，但比例很低。

- no_boxed 只有 `20` 个
- 这些样本平均长度却高达 `10453`
- `95%` 以 `Wait` 开头
- 只有 `5%` 正常闭合 `</think>`

这说明 4B 在极少数情况下也会出现和 `1.7B / L0H3` 类似的“反思死循环”，但频率远低得多，因此 `L0H22` 不是造成这种退化的主要瓶颈。

### 3.4 `outputs_4b`：6 个单题扫描

这部分不能作为正式 benchmark，但非常适合挖候选 head。

每个文件 top-1 strongest negative head：

| 文件 | 题目 id | top negative head | delta |
| --- | --- | --- | ---: |
| `wait_logit_1.csv` | `arithmetic_sum` | `L0H22` | -21.094 |
| `wait_logit_4.csv` | `combination_15_3` | `L0H1` | -4.719 |
| `wait_logit_5.csv` | `quadratic_sum_squares` | `L34H14` | -2.188 |
| `wait_logit_6.csv` | `triangle_area_13_20_21` | `L0H1` | -6.219 |
| `wait_loigit_2.csv` | `right_triangle_incircle` | `L34H14` | -2.625 |
| `wait_loigit_3.csv` | `four_digit_numbers` | `L34H14` | -2.375 |

如果只看 top-1 频率：

- `L34H14`：`3/6`
- `L0H1`：`2/6`
- `L0H22`：`1/6`

但更有价值的是 recurring top-k 分析。

反复出现在 top-10 中的 recurring head 包括：
L34H1、L34H14、L33H29、L0H1、L0H22

这比只盯着 top-1 更重要，因为它提示：

- 4B 上影响 reflective signal 的 head 很可能不是单个 head
- 更像是一组分布在 `17-35` 层之间的候选簇

## 4. 现有数据可以支持的机制性判断

### 4.1 `1.7B` 的 self-correction 更稀疏、更早层、更脆弱

证据：

- 全 head 扫描中最强 head 出现在 `layer 0`
- `layer 0` 的平均行为破坏显著高于其它层
- 单个 `L0H3` 就能把 corrected rate 从 `0.877` 打到 `0.072`

解释：

- `1.7B` 的自我纠错可能依赖少量早层 head 进行路由或状态切换
- 一旦关键 head 被屏蔽，模型不是变得“更固执”，而是直接失去稳定反思能力

### 4.2 `4B` 的 self-correction 更分布式、更稳健

证据：

- `L0H22` 对单题可很强，但对全量样本平均影响弱
- 4B 单题扫描中 recurring head 明显分布在中后层
- `L0H22` 被屏蔽后，大多数样本仍 corrected

解释：

- 4B 的 reflective computation 很可能是冗余实现的
- 单个 head 被屏蔽，通常只会削弱某个局部通路，而不是让整个自我纠错彻底失效

### 4.3 “反思关键词”与“有效反思”必须分开

证据：

- `1.7B / L0H3` 的 no_boxed 样本 keyword rate 仍然不低
- `4B / L0H22` 的 no_boxed 样本 keyword rate 甚至达到 `1.000`
- 但这些样本几乎不闭合 `</think>`，且长度异常长

解释：

- “Wait”“Hold on” 这类词出现，只说明表面样式进入了 reflective mode
- 是否形成有效纠错，还取决于：
  - 是否能闭合 `</think>`
  - 是否能回到结构化推理
  - 是否能收束到合法 `\\boxed{}`

### 4.4 wait-logit 更像“入口信号”，不是完整因果解释

证据：

- `outputs_1.7b` 中 wait-logit 与行为破坏相关性只有中等水平
- `4B / L0H22` 的 keep_wrong 样本并没有更负的 wait-logit delta，反而是正值

解释：

- wait-logit 适合做筛选信号
- 但真正决定纠错成败的，可能是后续一整段 reflective rollout 是否还能稳定展开并闭合
