# Qwen Self-Correction Head Ablation

This repository contains tools to analyze self-correction behavior in Qwen CoT continuation by ablating attention heads one by one.

Main entry script:
- `find_wait_head.py`

Main modules:
- `find_wait_head_lib/ablation.py`: head hook and head enumeration
- `find_wait_head_lib/pipeline.py`: stage1->tamper->analysis pipeline
- `find_wait_head_lib/model_utils.py`: model loading and generation
- `find_wait_head_lib/parallel_utils.py`: multi-GPU worker logic
- `find_wait_head_lib/io_utils.py`: dataset IO and summary/statistics writing

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
