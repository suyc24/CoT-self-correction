# Qwen Self-Correction Head Ablation

This repository contains tools to analyze self-correction behavior in Qwen CoT continuation by ablating attention heads one by one.

It now also includes a modular CoT research framework under `cot_research/` for systematic experiments on internal reasoning inside `<think>...</think>`.

Main entry script:
- `find_wait_head.py`

Main modules:
- `cot_research/head_ablation.py`: head hook and head enumeration for ablation experiments
- `cot_research/self_correction.py`: stage1->tamper->analysis pipeline for self-correction experiments
- `cot_research/model_utils.py`: model loading and generation helpers for the self-correction pipeline
- `cot_research/self_correction_parallel.py`: multi-GPU worker logic for self-correction ablations
- `cot_research/self_correction_io.py`: dataset IO and summary/statistics writing for self-correction runs

Modular framework:
- `cot_research/generation.py`: backend abstraction for HF and mock backends, stage1/full/continuation generation
- `cot_research/cot_editing.py`: registered think-edit strategies
- `cot_research/head_intervention.py`: registered attention-head interventions
- `cot_research/text_analysis.py`: keyword matching and repair-signal analysis
- `cot_research/token_analysis.py`: string/token occurrence counting and step-score extraction
- `cot_research/experiment_runner.py`: config-driven experiment runner
- `cot_research/summarize_results.py`: post-run aggregation

## Modular CoT Framework

Run the smoke test:

```bash
python -m cot_research.experiment_runner --config configs/cot_smoke_mock.json
```

Run a baseline HF experiment:

```bash
python -m cot_research.experiment_runner --config configs/cot_baseline_qwen_math.json
```

Run tamper + head ablation + keyword/token analysis:

```bash
python -m cot_research.experiment_runner --config configs/cot_tamper_ablate_analysis_qwen_math.json
```

Summarize an existing `rows.jsonl`:

```bash
python -m cot_research.summarize_results --input outputs/cot_research_tamper_ablate/rows.jsonl
```

## 1) Environment Setup

Recommended Python version: `3.10+`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

If you also want to run the evaluation toolkit under `evaluation/`:

```bash
pip install -r evaluation/requirements.txt
```

## 2) Dataset Format

For `find_wait_head.py`, input file should be JSONL with fields:
- required: `id`, `correct_answer`, `wrong_answer`
- recommended: `question`
- optional legacy: `prompt_prefix`

Default test file:
- `evaluation/data/self_correction_ablation/test_questions.jsonl`

## 3) Basic Run

### Single-machine run (default settings)

```bash
python find_wait_head.py \
  --model_name_or_path Qwen/Qwen3-4B \
  --input_jsonl evaluation/data/self_correction_ablation/test_questions.jsonl \
  --output_dir outputs/self_correction_full
```

### Multi-GPU head ablation

```bash
python find_wait_head.py \
  --parallel_heads \
  --parallel_gpu_ids 0,1,2,3 \
  --baseline_gpu_id 0 \
  --model_name_or_path Qwen/Qwen3-4B \
  --input_jsonl evaluation/data/self_correction_ablation/test_questions.jsonl \
  --output_dir outputs/self_correction_full
```

### Enable sampling (instead of greedy)

```bash
python find_wait_head.py --do_sample --temperature 1.4 --top_p 0.9
```

## 4) Key Outputs

In `--output_dir` (default `outputs/self_correction_full`):

- `ablation_no_reflect_wrong_only.jsonl`
  - filtered records of ablation runs
- `head_summary.csv`
  - per-head correction summary
- `run_config.json`
  - full run configuration
- `head_wait_token_logits.jsonl`
  - per-example per-head wait-token logits (`baseline` vs `ablated`)
- `head_wait_token_logit_ranking.csv`
  - head ranking by wait-token logit delta magnitude
- `wait_logit_by_example/*.csv`
  - one CSV per question (`example_id`), each containing all ablated heads for that question

## 5) Useful Arguments

- `--head_spec "L0H0,L1H3"`: run only selected heads
- `--max_examples N`: limit sample count
- `--print_cot / --no-print_cot`: print or silence CoT in terminal
- `--wait_token_text "Wait"` or `--wait_token_id <id>`: wait-token logit tracking target
- `--local_files_only / --no-local_files_only`: force local cache or allow remote fetch

## 6) Notes

- The script can run with `device_map=auto` and supports distributed module placement.
- For very large models, tune `--max_stage1_tokens` and `--max_new_tokens` to avoid OOM.
- Test-only smoke outputs were cleaned from `outputs/`; non-smoke outputs are kept.
