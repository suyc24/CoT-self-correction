# AGENT_GUIDE

This repository is an automated research workspace for mechanistic interpretability and chain-of-thought (CoT) experiments on Qwen-family models, especially `<think>...</think>` reasoning.

## 1. Mission

This repo supports fast, modular, repeatable experiments on:
- CoT generation and continuation
- self-correction / reflection signals
- attention-head interventions and ablations
- attention pattern analysis
- repetition / looping behavior
- accuracy under interventions
- comparisons across model sizes and variants

Internal reasoning behavior is a first-class object of study, not just final answers.

## 2. Core Rules

Optimize for, in order:
1. experimental correctness
2. reusability and modularity
3. throughput and automation
4. result traceability
5. minimal duplicated logic

Default preferences:
- put reusable logic in `cot_research/`
- keep `scripts/` thin
- save explicit metadata
- preserve partial outputs
- do not overwrite unrelated work
- avoid destructive git commands

## 3. Repository Structure

### 3.1 Reusable code

All reusable experiment logic should live in `cot_research/`.

Key modules include:
- `generation.py`
- `prompt_utils.py`
- `datasets.py`
- `row_utils.py`
- `head_intervention.py`
- `head_ablation.py`
- `text_analysis.py`
- `repetition_analysis.py`
- `cot_accuracy.py`
- `attention_sink_analysis.py`
- `head_attention_pattern.py`
- `local_attention_analysis.py`
- `summary_utils.py`, `io_utils.py`, `runtime_utils.py`

Do not introduce new reusable logic outside `cot_research/`.

### 3.2 Scripts

`scripts/` should be thin entrypoints:
- parse args
- select data
- call library code
- write outputs

If logic is repeated across scripts, move it into `cot_research/`.

### 3.3 Experiment archive layout

Kept experiments should live under:
- `experiment_results/experiments/<experiment_id>/`

Current phase families include:
- `phase1_wait_head_discovery`
- `phase2_l0h22_scale_4b`
- `phase3_l0h3_mechanism`
- `phase4_scale_benchmark`
- `phase5_head_locality`

Prefer adding new kept experiments under the closest existing phase family when appropriate.

## 4. Backend Policy

Use the backend that matches the experiment.

### Prefer vLLM for:
- high-throughput generation
- baseline CoT collection
- large sweeps without hooks or attention extraction

### Prefer HF / Transformers for:
- `output_attentions=True`
- per-head ablation or scaling
- exact internal logits / tensors
- intervention on attention outputs
- query-position-specific attention analysis

Do not force vLLM into experiments that need internal tensors.

## 5. Research Conventions

### 5.1 CoT-first policy

Whenever possible, save and analyze:
- full generated text
- continuation text
- `<think>` text or continuation-only think text
- final boxed answer
- generation length

### 5.2 Reflection analysis

Do not treat `Wait` as the only reflection signal.

Prefer configurable reflection lexicons, including:
- `wait`
- `hold on`
- `let me check`
- `let me think`
- `actually`
- `on second thought`
- relevant Chinese reflection phrases when needed

### 5.3 Repetition detection

All repetition judgments must go through:
- `cot_research.repetition_analysis`

Do not reimplement repetition heuristics ad hoc in scripts.

### 5.4 Accuracy judging

All answer judging should use:
- `cot_research.cot_accuracy`

This keeps offline judging and cross-run comparison consistent.

## 6. Experiment Design Rules

### 6.1 Estimate size before launch

Before remote execution, estimate:
- number of examples
- number of heads
- number of scales / conditions
- max generation length
- whether generation repeats per head

Prefer a pilot before a full sweep.

Typical sizing:
- small pilot: `8`, `12`, `50`, `100`
- medium run: `200` to `1000`
- full sweeps only after signal is validated

### 6.2 Keep compared examples fixed

When comparing interventions or scales, use the same sampled examples across conditions.

### 6.3 Keep outputs focused

Do not generate or save large extra artifacts unless they are needed.

### 6.4 Save reproducibility metadata

Each experiment output directory should contain at least:
- `run_config.json`
- per-example rows (`jsonl` or `csv`)
- one or more summaries (`json`, `csv`, or `md`)

## 7. Remote Execution Policy

### 7.1 Default remote environment

When the user asks to run on the server, use:

- host: `101.6.96.183`
- username: `yucheng`
- SSH ports: `8002` to `8006`
- primary login: `ssh -p 8002 yucheng@101.6.96.183`
- remote repo root: `/home/yucheng/experiment/Qwen2.5-Math`
- micromamba binary: `/home/yucheng/bin/micromamba`
- micromamba root: `/home/yucheng/micromamba`
- default env: `qwen_math`
- long-running jobs: `tmux`

Typical activation:
```bash
eval "$(/home/yucheng/bin/micromamba shell hook -s bash)"
micromamba activate qwen_math
````

Do not store authentication material in this guide.

### 7.2 GPU policy

Use only GPU ids explicitly allowed by the user.

### 7.3 Long-running jobs

For remote jobs:

* create a dedicated `tmux` session
* log stdout/stderr under `logs/`
* verify the job really started
* verify GPU / worker activity
* stop after confirming run health unless the user asked for more

A run counts as started only if at least one is true:

* main loop has started
* progress bars are moving
* output files are being written
* GPU processes / memory show actual execution

### 7.4 Smoke test first for new scripts

Before launching a full remote run for a new script:

1. run `py_compile`
2. run `--help`
3. run a minimal smoke test
4. only then launch the full job

## 8. Sync Policy

Use `rsync` for code and result transfer.

### Upload

* sync only touched files when possible
* do not upload the whole repo blindly

### Download

* preserve summaries and aggregate outputs
* avoid pulling unnecessary raw dumps if not needed
* place pulled results under `experiment_results/`

### Secrets

Treat remote credentials and paths as local operational metadata.
Do not copy or publish them outside the trusted workspace unless explicitly asked.

## 9. Output Organization

### 9.1 Temporary outputs

Active development outputs may live under:

* `outputs/`

Treat `outputs/` as temporary, not the final home for kept runs.

### 9.2 Canonical experiment folder

Every kept experiment should have one canonical folder:

* `experiment_results/experiments/<experiment_id>/`

That folder should contain:

* brief / plan / tracker
* launch command or wrapper
* pulled summaries and row-level outputs
* review notes and final report

Do not leave a kept experiment scattered across unrelated folders.

### 9.3 Naming conventions

Experiment directories should encode:

* task family
* model
* main intervention or head
* date or run id

Examples:

* `outputs/head_attention_pattern/qwen3_1p7b_l0h3_20260404_1`
* `outputs/repetition/l0h3_suppression_qwen3_1p7b`

## 10. Interpretation Guidance

When reporting results, distinguish carefully between:

* prefix attention vs local-neighbor attention
* self-loop attention vs true local-window attention
* reduced repetition vs improved correctness
* reduced reflection count vs reduced reflection density
* base vs instruct model behavior

Do not overclaim from a small pilot.

## 11. ARIS Integration

If the user asks for ARIS / ARIS-style / Auto-claude-code-research-in-sleep, treat ARIS as the high-level orchestration layer.

In this repo:

* use ARIS for workflow staging and review loops
* use this guide for repo-local implementation, experiment sizing, remote execution, output layout, and interpretation
* do not reinstall or reconfigure ARIS unless the user explicitly asks
* if ARIS behavior conflicts with repo-specific experimental constraints, follow this guide unless the user explicitly overrides it
* if the user asks for only part of the ARIS workflow, stop at that stage

## 12. Standard Workflow

For a new experiment, default to:

1. clarify the scientific question
2. identify whether the task is:

   * pure generation
   * intervention / ablation
   * attention extraction
   * offline analysis
3. implement reusable logic in `cot_research/`
4. create or update a thin script in `scripts/`
5. run:

   * `py_compile`
   * `--help`
6. run a smoke test
7. if remote is needed:

   * sync with `rsync`
   * start in `tmux`
   * verify startup
8. pull results back with `rsync`
9. consolidate into `experiment_results/experiments/<experiment_id>/`
10. summarize results in plain language with direct file references

## 13. Current Important Entry Points

Useful scripts include:

* `find_wait_head.py`
* `scripts/analyze_attention_sink_heads.py`
* `scripts/analyze_head_attention_pattern.py`
* `scripts/analyze_local_attention_heads.py`
* `scripts/analyze_next_token_current_heads.py`
* `scripts/classify_attention_locality.py`
* `scripts/run_boundary_head_probe.py`
* `scripts/find_reflection_heads_by_wait_ablation.py`
* `scripts/test_l0h3_repetition_suppression.py`
* `scripts/test_head_boost_effects.py`
* `scripts/run_l0h3_scale_wait_length.py`
* `scripts/run_l0h3_copy_suppression.py`
* `scripts/run_phrase_copy_mechanism.py`
* `scripts/run_local_copy_temptation.py`
* `scripts/run_collapse_prefix_mechanism.py`
* `scripts/run_repetition_causal_case_study.py`
* `scripts/run_repetition_scale_logit_sweep.py`
* `scripts/run_scale_correctness_benchmarks.py`
* `scripts/discover_l0h3_like_heads.py`
* `scripts/filter_repetition_cot.py`
* `scripts/judge_cot_accuracy.py`
* `scripts/collect_qwen3_numinamath_cot.py`

## 14. Default Assumptions

Unless the user says otherwise:

* prioritize `cot_research` reuse
* prefer small pilots before large sweeps
* use `rsync` for remote transfer
* use `tmux` for long runs
* preserve partial outputs
* favor interpretable summaries over raw dumps

## 15. What Good Looks Like

A good contribution usually has:

* a clear scientific question
* reusable implementation in `cot_research/`
* a thin entry script in `scripts/`
* a smoke-tested execution path
* reproducible configs and saved summaries
* outputs that answer the actual user question

