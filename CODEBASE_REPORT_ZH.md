# Qwen2.5-Math 仓库代码说明报告

## 1. 先说结论：这个仓库为什么会让人看晕

这个仓库不是单一路径的“小实验脚本集合”，而是两套思路叠在一起：

1. 一套是较早的 self-correction / head ablation 流水线。
2. 一套是后来抽出来的 `cot_research/` 模块化框架。
3. `scripts/` 下面的脚本并不都走同一个统一入口，有些直接复用 `cot_research/` 的公共模块，有些还保留了比较“手工拼装”的实验流程。

所以你现在看到的感觉不是“这仓库没结构”，而是“它有结构，但结构分成了旧路线和新路线两层”。

可以先把它理解成下面这张图：

```text
数据/已有样本
  -> prompt 组织
  -> 模型后端（HF 或 vLLM）
  -> 可选编辑 / 可选 head intervention / 可选 head ablation
  -> 生成
  -> 文本分析 / token分析 / repetition分析 / 准确率分析 / attention分析
  -> 汇总输出
```

其中 `cot_research/` 就是把这条链拆成了一堆可复用模块。

## 2. 这个仓库里最重要的两条代码路线

### 2.1 新路线：模块化 CoT research framework

这条路线的核心是：

- `cot_research/generation.py`
- `cot_research/experiment_runner.py`
- `cot_research/schemas.py`
- `cot_research/summary_utils.py`

这套路线的目标是把实验拆成：

- 数据加载
- prompt 构造
- 后端选择
- 编辑或干预
- 生成
- 分析
- 汇总

优点是可复用、组件边界更清楚。

### 2.2 旧路线：self-correction / wait-head / ablation

这条路线的核心是：

- `cot_research/self_correction.py`
- `cot_research/self_correction_parallel.py`
- `cot_research/self_correction_io.py`
- `cot_research/model_utils.py`
- `cot_research/head_ablation.py`

这套路线更像是为“先生成 stage1 CoT，再篡改答案，再做 ablation，看模型会不会自我纠错”这一类实验服务的专用流水线。

优点是针对性强，缺点是和新框架有一定重叠。

### 2.3 当前已经开始统一的共享底座

这次整理之后，可以把两条路线共同依赖的底层能力先理解成一层“共享底座”：

- `cot_research/model_utils.py`

它现在不再只是旧 self-correction 路线的模型工具，而是同时承接了几类公共能力：

- HF 模型与 tokenizer 加载
- stop string 对应的 stopping criteria
- 输入张量放到正确 device
- next-token logit 与 continuation 生成
- `output_attentions=True` 的公共前向
- decoder layer / attention module / head shape 的解析
- head label 解析

也就是说，新旧路线虽然还没有完全合成一个统一入口，但已经开始共享同一套底层 HF / attention 公共函数了。

## 3. `cot_research/` 各模块做什么

下面按功能分组讲，不按文件名字母顺序讲。

### 3.1 入口与导出层

#### `cot_research/__main__.py`

- 作用：让你可以直接用 `python -m cot_research` 运行默认入口。
- 实际上它只是转发到 `experiment_runner.main()`。

#### `cot_research/__init__.py`

- 作用：把包里最常用的函数和类重新导出。
- 你可以把它理解成这个包的“公共 API 索引页”。

### 3.2 配置、数据结构、注册器

#### `cot_research/schemas.py`

- 作用：定义整个框架里反复传来传去的数据结构。
- 关键类：
- `DatasetExample`：统一的数据样本格式。
- `GenerationConfig`：生成参数。
- `BackendConfig`：后端参数。
- `GenerationResult`：一次生成的结果。
- `EditOperation`、`TokenCountResult`、`InterventionResult`：编辑、token分析、干预分析的结构化输出。

如果你想知道“一个实验脚本最后到底在传什么对象”，先看这个文件很有帮助。

#### `cot_research/registry.py`

- 作用：提供一个很轻量的注册表。
- 主要被两个地方使用：
- `cot_editing.py` 里的编辑器注册。
- `head_intervention.py` 里的 intervention 注册。

也就是说，这个文件是在支撑“名字 -> 实现函数”的插件式写法。

### 3.3 Prompt、后端、模型加载

#### `cot_research/prompt_utils.py`

- 作用：把 question 转成聊天模型真正吃的 prompt。
- 关键点：
- 优先用 tokenizer 的 `apply_chat_template`。
- 如果模型支持 `enable_thinking`，这里会把它传进去。
- 如果模板失败，会回退到一个朴素的 `System/User/Assistant` 文本模板。

#### `cot_research/generation.py`

- 作用：这是新框架里最核心的抽象层，统一“怎么生成”。
- 它定义了一个抽象基类 `GenerationBackend`，然后实现了三种后端：
- `MockGenerationBackend`：给 smoke test 用。
- `HFGenerationBackend`：基于 Hugging Face / Transformers。
- `VLLMGenerationBackend`：基于 vLLM。

这个文件解决了几个关键问题：

- 统一 prompt 编码 / 解码接口。
- 统一单条生成和批量生成接口。
- 统一 stop string 处理。
- 在 HF 后端里支持 step scores 和 next-token stats。
- 在 `create_backend()` 里根据 `hf` / `vllm` / `auto` 选择后端。

这个文件是“现代框架”的中心。

#### `cot_research/model_utils.py`

- 作用：这是旧路线里更底层的 HF 模型工具箱。
- 主要能力：
- 加载 HF 模型和 tokenizer。
- 带重试策略的模型加载。
- 给定 prompt 直接生成 continuation。
- 读取某个目标 token 的 next-token logit。

和 `generation.py` 的关系可以理解为：

- `generation.py` 更像新框架的统一接口层。
- `model_utils.py` 更像旧 self-correction 路线的底层工具层。

### 3.4 数据、行结构、IO、运行时工具

#### `cot_research/datasets.py`

- 作用：负责读取数据和做基础数据适配。
- 主要事情：
- 校验本地 JSONL 是否满足最小字段要求。
- 加载本地样本到 `DatasetExample`。
- 从 `AI-MO/NuminaMath-CoT` 读取数据集。
- 把 NuminaMath 的原始 row 转成这个仓库自己的统一格式。

#### `cot_research/row_utils.py`

- 作用：这是“单行样本怎么理解”的公共工具。
- 它解决的问题非常实际：
- 从很多可能的字段里挑 continuation 文本。
- 从不同字段里提取 example id。
- 从 `problem/messages` 里构造 question。
- 尝试从已有字段恢复 prompt prefix。

很多脚本都在用它，因为实验输入数据的字段风格不完全统一。

#### `cot_research/io_utils.py`

- 作用：最基础的文件读写工具。
- 提供：
- `load_jsonl`
- `dump_jsonl`
- `peek_first_json_row`
- `write_csv`
- `write_json`
- `truncate_text`

它不复杂，但几乎所有脚本都会用到。

#### `cot_research/runtime_utils.py`

- 作用：运行时辅助工具。
- 主要包括：
- 解析 GPU id 列表。
- 按 contiguous 方式切分样本。
- 根据 worker/gpu 解析 device map。
- 随机种子设置。

这类函数在多 GPU 脚本里很常见。

### 3.5 文本、token、答案、准确率分析

#### `cot_research/answer_extraction.py`

- 作用：从输出文本里抽最终答案，并做比较。
- 主要逻辑：
- `extract_last_boxed`：取最后一个 `\boxed{...}`。
- `normalize_answer`：做去空格、去模板词、去逗号等标准化。
- `answers_match`：文本比对失败时再尝试做简单数值比对。
- `classify_outcome`：把结果分成 `corrected / keep_wrong / other_answer / no_boxed_answer`。

#### `cot_research/text_analysis.py`

- 作用：围绕 `<think>...</think>` 文本做关键词和 repair-signal 分析。
- 主要能力：
- 抽取 think segment。
- 抽取 continuation 里的 think 文本。
- 统计关键词命中和位置。
- 计算 repair signal 的弱/中/强。

这是“reflection / self-correction 语言信号”分析的基础模块。

#### `cot_research/token_analysis.py`

- 作用：统计某些目标字符串或 token 序列在 full_text / think_text 中出现了几次。
- 还能把 step-level token scores 里和目标 token 有关的部分筛出来。

适合做“Wait 出现了几次”“某个 token 在思维链里是不是更活跃”之类的分析。

#### `cot_research/repetition_analysis.py`

- 作用：统一的重复/死循环检测模块。
- 它会综合多种证据：
- 相同 token 长跑。
- 尾部 token n-gram 循环。
- 重复行。
- 尾部词级重复。

然后输出：

- 是否判定为 repetitive。
- 触发了哪些规则。
- 分数和详细 profile。

这是整个仓库里“repetition 检测”的标准入口。

#### `cot_research/cot_accuracy.py`

- 作用：统一准确率判断逻辑。
- 它支持两种场景：
- 单条输出是否答对。
- baseline vs intervention 的答案变化比较。

常见功能：

- 从 row 里提取最终答案。
- 从 row 或 gold row 里提取 gold answer。
- 生成 `correct / wrong / unverifiable` 或 `newly_correct / regressed / remained_*` 之类标签。
- 汇总 accuracy summary。

### 3.6 汇总与后处理

#### `cot_research/summary_utils.py`

- 作用：把逐条 row 聚合成 summary 表。
- 主要有三类汇总：
- condition 级别汇总。
- token target 汇总。
- next-token target 汇总。

你可以把它理解成“新框架里通用的 summary builder”。

#### `cot_research/summarize_results.py`

- 作用：一个很薄的 CLI。
- 输入一个 `rows.jsonl`，自动调用 `summary_utils.py` 输出：
- `summary.csv`
- `token_target_summary.csv`
- `next_token_summary.csv`
- `summary.json`

### 3.7 编辑与干预

#### `cot_research/cot_editing.py`

- 作用：对 CoT 文本做“编辑”。
- 它不是改模型参数，而是改已经生成好的文本前缀。
- 主要编辑器包括：
- 替换最后一个 boxed answer。
- 替换尾部答案。
- 替换关键词。
- 替换任意 span。
- 追加错误 boxed answer。
- `auto_tamper_answer` 自动选择合适的篡改方式。

这个模块常用于“先让模型写一段思路，再故意把答案篡改，然后让它继续生成”的实验。

#### `cot_research/head_ablation.py`

- 作用：做“head ablation”，也就是把某个 head 在 `o_proj` 前对应的 slice 置零。
- 核心对象：
- `HeadSpec`
- `SingleHeadAblationHook`
- `MultiHeadAblationHookSet`

它更偏“删掉某个头的贡献”。

#### `cot_research/head_intervention.py`

- 作用：做更一般的 head intervention。
- 和 `head_ablation.py` 的区别是：
- `head_ablation.py` 主要是置零。
- `head_intervention.py` 允许 scale、identity、zero 等更通用的操作。

核心对象：

- `HeadTarget`
- `LayerHeadInterventionHook`
- `MultiLayerHeadIntervention`
- `INTERVENTION_REGISTRY`

如果你做的是 `L0H3 x 1.2` 这种实验，核心就是这个模块。

### 3.8 注意力分析模块

#### `cot_research/local_attention_analysis.py`

- 作用：分析某个 query 位置对 prefix 或局部位置的注意力分布。
- 典型场景：
- 找到 `</think>` 前那个 query token。
- 看各个 head 对前 k 个 token 的注意力质量。
- 把 prefix 注意力 mask 掉以后，看注意力向量变化有多大。

它更像“prefix/local attention probe”的分析库。

#### `cot_research/head_attention_pattern.py`

- 作用：分析单个 head 的注意力分布形状。
- 典型指标：
- entropy
- max value
- 对前 1/4/8/... 个 token 的累计质量
- 各距离 bucket 的注意力质量
- top-k attended positions

它回答的问题更像：

- 这个头更关注自己、近邻、还是长距离历史？

#### `cot_research/attention_sink_analysis.py`

- 作用：分析 attention sink 现象。
- 典型问题：
- 某个 head 是否总把大量注意力打到序列最前面几个 token？
- 它对“前缀 token / sink token / position 0”的注意力是不是异常高？

这个模块会输出：

- 每个 head 的 sink mass。
- 在每个 example 内的 rank。
- 跨样本的汇总统计。

### 3.9 旧版 self-correction 流水线模块

#### `cot_research/self_correction.py`

- 作用：旧版两阶段自纠错实验的核心逻辑。
- 典型流程：
- 先生成 stage1 `<think>...</think>`。
- 在 think 里把正确答案替换成错误答案，或者补一个错的 boxed answer。
- 再把篡改后的前缀喂回模型继续生成。
- 分析模型最后有没有纠正回来。

这里面的 `prepare_example_prefix()` 和 `analyze_generation()` 是旧流水线的关键函数。

#### `cot_research/self_correction_parallel.py`

- 作用：把旧 self-correction 实验拆成多 GPU worker。
- 一个 worker 负责：
- 做 baseline prepare + baseline generation。
- 或者做一批 head ablation 运行。

它本质上是旧流程的并行执行器。

#### `cot_research/self_correction_io.py`

- 作用：给旧 self-correction 流水线配套的 IO 和 summary。
- 负责：
- 样本校验。
- summary 统计。
- prepared examples 的存取。
- wait logit 统计与导出。

这个模块不是通用框架层，而是旧实验线的专用后处理层。

## 4. `scripts/` 里每个脚本在组合哪些代码

这一部分是最实用的。你以后看到某个脚本，不用再自己顺着 import 一行一行猜。

### 4.1 纯生成 / 数据收集 / 后处理类

#### `scripts/collect_qwen3_numinamath_cot.py`

- 作用：用 `vLLM` 从 NuminaMath 批量收集 Qwen CoT。
- 组合的模块：
- `datasets.py`：读 NuminaMath。
- `prompt_utils.py`：把题目转成 chat prompt。
- `row_utils.py`：构造 problem / reference solution。
- `text_analysis.py`：抽 think 文本。
- `repetition_analysis.py`：做基础 repetition 特征统计。
- `answer_extraction.py`：抽 boxed answer。
- `runtime_utils.py`：解析多 GPU。
- 关键特点：
- 这是仓库里最明显的 `vLLM` 脚本。
- 适合大批量纯生成，不做 head-level 内部干预。

#### `scripts/filter_repetition_cot.py`

- 作用：从已有 CoT JSONL 里筛出明显死循环的样本。
- 组合的模块：
- `io_utils.py`
- `repetition_analysis.py`
- `row_utils.py`
- 输出：
- 重复样本 JSONL
- summary JSON
- Markdown 预览

#### `scripts/judge_cot_accuracy.py`

- 作用：给已经生成好的 JSONL 批量补准确率判断。
- 组合的模块：
- `cot_accuracy.py`
- `io_utils.py`
- `summary_utils.py`
- 支持：
- single-condition rows
- comparison rows

#### `scripts/summarize_head_boost_worker_outputs.py`

- 作用：如果 `test_head_boost_effects.py` 中途断了，这个脚本可以直接把 worker 的零散输出重新拼成最终 summary。
- 组合的模块：
- `cot_accuracy.py`
- `io_utils.py`
- 本质：
- 它不是实验本身，而是 head boost 实验的“恢复脚本”。

#### `scripts/smoke_test_cot_framework.sh`

- 作用：跑一个最小 mock 配置，验证 `cot_research.experiment_runner` 这条新框架链路能不能正常工作。
- 组合的模块：
- 实际上就是调用 `python -m cot_research.experiment_runner --config configs/cot_smoke_mock.json`

### 4.2 头部干预 / ablation / 比较实验类

#### `scripts/test_l0h3_repetition_suppression.py`

- 作用：测试对 `L0H3` 做 intervention 后，重复现象会不会减轻。
- 默认流程：
- 读已经被判成 repetitive 的样本。
- 重新跑 baseline。
- 对同样 prompt 施加 `L0H3` intervention。
- 比较 repetition 和 accuracy。
- 组合的模块：
- `generation.py`：生成。
- `head_intervention.py`：做 zero/scale 等 intervention。
- `repetition_analysis.py`：判断 suppression / induced repetition。
- `cot_accuracy.py`：比较 baseline 与 intervention 的答案变化。
- `row_utils.py`：恢复 prompt。
- `runtime_utils.py`：多 GPU 分片。
- `summary_utils.py`：导出 summary。

#### `scripts/test_head_boost_effects.py`

- 作用：更一般地比较“boost 一组 heads”对准确率和 repetition 的影响。
- 和上一个脚本的关系：
- 它是一个更通用、更大的版本。
- `test_l0h3_repetition_suppression.py` 可以看成这个思路的特化版。
- 组合的模块：
- `datasets.py`：可直接从 NuminaMath 取样。
- `generation.py`
- `head_intervention.py`
- `repetition_analysis.py`
- `cot_accuracy.py`
- `summary_utils.py`
- `runtime_utils.py`
- 特点：
- 支持 baseline rerun 或 stored baseline。
- 支持多 head、不同 intervention 类型。

#### `scripts/run_l0h3_scale_wait_length.py`

- 作用：这是你最近用到的脚本，专门比较 `L0H3` 不同 scale 对：
- reasoning 长度
- reflection 频率
- repetition
- answer correctness

- 组合的模块：
- `datasets.py`：读取本地 JSONL 或 NuminaMath。
- `generation.py`：HF 后端生成。
- `head_intervention.py`：对 `L0H3` 做 scale。
- `text_analysis.py`：统计 reflection 关键词。
- `repetition_analysis.py`：分析重复。
- `cot_accuracy.py`：判断正确率。
- `answer_extraction.py`：抽答案。
- `row_utils.py`：恢复 prompt。
- `runtime_utils.py`：多 GPU 切任务。
- `io_utils.py`：保存 `rows.csv` / `summary.csv` / `summary.json`。

- 这是一个很典型的“组合式脚本”：它自己不实现底层逻辑，而是把好几个模块串起来。

#### `scripts/find_reflection_heads_by_wait_ablation.py`

- 作用：找“和 wait / reflection 信号有关的头”。
- 核心实验想法：
- 先生成 stage1。
- 找到 think 里 `wait` 的位置。
- 取 `wait` 前缀作为 anchor。
- 测 baseline 的 wait token logit。
- 对不同 head 做 ablation，看 wait token logit 如何变化。

- 组合的模块：
- `generation.py`
- `head_ablation.py`
- `text_analysis.py`
- `row_utils.py`
- `runtime_utils.py`
- `io_utils.py`

- 这是一个“由 wait 这个语言信号反推可疑 heads”的脚本。

### 4.3 注意力分析类

#### `scripts/analyze_attention_sink_heads.py`

- 作用：分析哪些 head 像 attention sink head。
- 核心实验逻辑：
- 选一段分析文本。
- 前向拿 `output_attentions=True`。
- 定义 sink positions。
- 用 `attention_sink_analysis.py` 统计每个 head 的 sink mass。

- 组合的模块：
- `attention_sink_analysis.py`
- `head_intervention.py` 里的 `list_model_heads`
- `prompt_utils.py`
- `row_utils.py`
- `runtime_utils.py`
- `summary_utils.py`

#### `scripts/analyze_local_attention_heads.py`

- 作用：分析在最终 `</think>` 前那个 query 位置上，各个 head 是否偏向 prefix/local attention。
- 核心实验逻辑：
- 先生成到 `</think>`。
- 只保留在关闭前已经有 boxed answer 的样本。
- 重新前向拿 attentions。
- 在 `</think>` 前一位 query 上，统计 prefix mass 和 mask 后的向量变化。

- 组合的模块：
- `generation.py`
- `local_attention_analysis.py`
- `answer_extraction.py`
- `row_utils.py`
- `runtime_utils.py`
- `io_utils.py`

#### `scripts/analyze_head_attention_pattern.py`

- 作用：对一个指定 head 做更细的注意力形状分析。
- 和 `analyze_local_attention_heads.py` 的区别：
- `analyze_local_attention_heads.py` 是“所有头一起排名，关注 prefix/local 指标”。
- 这个脚本是“盯住一个头，看它的距离分布、top positions、entropy 等模式”。

- 组合的模块：
- `generation.py`
- `head_attention_pattern.py`
- `local_attention_analysis.py`：用于找 close-think query
- `answer_extraction.py`
- `row_utils.py`
- `runtime_utils.py`
- `io_utils.py`

## 5. 哪些脚本更像“实验”，哪些更像“工具”

### 5.1 更像正式实验脚本

- `collect_qwen3_numinamath_cot.py`
- `test_l0h3_repetition_suppression.py`
- `test_head_boost_effects.py`
- `run_l0h3_scale_wait_length.py`
- `find_reflection_heads_by_wait_ablation.py`
- `analyze_attention_sink_heads.py`
- `analyze_local_attention_heads.py`
- `analyze_head_attention_pattern.py`

### 5.2 更像后处理 / 辅助工具

- `filter_repetition_cot.py`
- `judge_cot_accuracy.py`
- `summarize_head_boost_worker_outputs.py`
- `smoke_test_cot_framework.sh`

## 6. 如果按“读懂仓库”的顺序，建议你这样看

### 6.1 第一层：先看通用骨架

建议顺序：

1. `cot_research/schemas.py`
2. `cot_research/prompt_utils.py`
3. `cot_research/generation.py`
4. `cot_research/row_utils.py`
5. `cot_research/io_utils.py`

这一层看完，你会知道：

- 一个样本长什么样
- prompt 怎么构造
- 后端怎么跑
- 输出怎么保存

### 6.2 第二层：再看几个分析模块

建议顺序：

1. `cot_research/answer_extraction.py`
2. `cot_research/text_analysis.py`
3. `cot_research/repetition_analysis.py`
4. `cot_research/cot_accuracy.py`
5. `cot_research/summary_utils.py`

这一层看完，你会知道：

- 仓库怎么定义“reflection”
- 怎么定义“重复”
- 怎么定义“正确”
- summary 是怎么来的

### 6.3 第三层：看干预和 attention 模块

建议顺序：

1. `cot_research/head_intervention.py`
2. `cot_research/head_ablation.py`
3. `cot_research/local_attention_analysis.py`
4. `cot_research/head_attention_pattern.py`
5. `cot_research/attention_sink_analysis.py`

这一层看完，你就能大致明白：

- 仓库到底是怎么“动某个 head”的
- attention 类分析到底在算什么

### 6.4 第四层：最后再看具体脚本

如果你最近最关心的是你现在正在跑的实验，建议先看：

1. `scripts/run_l0h3_scale_wait_length.py`
2. `cot_research/head_intervention.py`
3. `cot_research/text_analysis.py`
4. `cot_research/repetition_analysis.py`
5. `cot_research/cot_accuracy.py`

如果你更关心 attention probe，建议先看：

1. `scripts/analyze_local_attention_heads.py`
2. `cot_research/local_attention_analysis.py`
3. `scripts/analyze_head_attention_pattern.py`
4. `cot_research/head_attention_pattern.py`

## 7. 你现在最需要记住的几个“总开关”

如果你只想先抓住仓库的主线，不想被所有文件淹没，可以先记住下面几个总开关：

- `generation.py`：决定模型怎么跑。
- `head_intervention.py`：决定 head 怎么改。
- `repetition_analysis.py`：决定什么叫重复。
- `cot_accuracy.py`：决定什么叫答对。
- `summary_utils.py`：决定最后 summary 怎么出。
- `run_l0h3_scale_wait_length.py`：把上面这些真正拼成一个实验。

## 8. 一句话总结

这个仓库本质上是一个“围绕 Qwen 推理链 `<think>...</think>` 做生成、干预、attention 分析、重复检测和准确率比较”的研究工作台。

`cot_research/` 负责提供可复用零件，`scripts/` 负责把这些零件拼成具体实验。之所以看起来复杂，不是因为每个文件都很难，而是因为它同时保留了旧 self-correction 流水线和新模块化框架两种组织方式。
