from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .answer_extraction import classify_outcome, extract_last_boxed
from .cot_editing import apply_editor_chain
from .datasets import dump_jsonl, load_examples
from .generation import create_backend
from .head_intervention import INTERVENTION_REGISTRY, MultiLayerHeadIntervention, resolve_head_targets
from .schemas import BackendConfig, DatasetExample, GenerationConfig
from .summary_utils import (
    summarize_condition_rows,
    summarize_next_token_targets,
    summarize_token_targets,
    write_csv,
    write_json,
)
from .text_analysis import (
    DEFAULT_REPAIR_LEXICON,
    analyze_repair_signals,
    analyze_text_keywords,
    extract_continuation_think_text,
    extract_think_segments,
)
from .token_analysis import analyze_token_targets


DEFAULT_KEYWORDS = [
    "wait",
    "no",
    "actually",
    "however",
    "mistake",
    "incorrect",
    "等一下",
    "不对",
    "重新检查",
    "重算",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Modular CoT research runner")
    parser.add_argument("--config", required=True, help="JSON config path.")
    parser.add_argument("--output_dir", default="", help="Optional override for output directory.")
    parser.add_argument("--max_examples", type=int, default=-1, help="Optional override for max examples.")
    parser.add_argument("--model_name_or_path", default="", help="Optional override for HF backend.")
    parser.add_argument("--device_map", default="", help="Optional override for HF backend device_map.")
    return parser.parse_args()


def _load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _make_generation_config(config: Dict[str, Any]) -> GenerationConfig:
    generation = dict(config.get("generation", {}))
    return GenerationConfig(
        system_prompt=str(generation.get("system_prompt") or "Please reason step by step in <think>...</think>."),
        assistant_prefix=str(generation.get("assistant_prefix", "<think>\n")),
        stage1_stop_string=str(generation.get("stage1_stop_string", "</think>")),
        max_stage1_tokens=int(generation.get("max_stage1_tokens", 2048)),
        max_new_tokens=int(generation.get("max_new_tokens", 1024)),
        do_sample=bool(generation.get("do_sample", False)),
        temperature=float(generation.get("temperature", 0.7)),
        top_p=float(generation.get("top_p", 0.9)),
        enable_thinking=bool(generation.get("enable_thinking", True)),
        capture_step_scores=bool(generation.get("capture_step_scores", False)),
    )


def _make_backend_config(config: Dict[str, Any], args: argparse.Namespace) -> BackendConfig:
    backend = dict(config.get("backend", {}))
    return BackendConfig(
        backend_type=str(backend.get("type", "auto")),
        model_name_or_path=str(args.model_name_or_path or backend.get("model_name_or_path", "Qwen/Qwen3-4B")),
        device_map=args.device_map or backend.get("device_map", "auto"),
        load_in_half=bool(backend.get("load_in_half", True)),
        use_fast_tokenizer=bool(backend.get("use_fast_tokenizer", False)),
        use_safetensors=bool(backend.get("use_safetensors", True)),
        local_files_only=bool(backend.get("local_files_only", True)),
        attn_implementation=str(backend.get("attn_implementation", "")),
        tensor_parallel_size=int(backend.get("tensor_parallel_size", 1)),
        gpu_memory_utilization=float(backend.get("gpu_memory_utilization", 0.9)),
        max_num_seqs=int(backend.get("max_num_seqs", 256)),
        max_num_batched_tokens=None if backend.get("max_num_batched_tokens") is None else int(backend.get("max_num_batched_tokens")),
        enable_chunked_prefill=bool(backend.get("enable_chunked_prefill", True)),
        enforce_eager=bool(backend.get("enforce_eager", False)),
        disable_log_stats=bool(backend.get("disable_log_stats", True)),
        mock_responses=dict(backend.get("mock_responses", {})),
    )


def _resolve_template(value: Any, context: Dict[str, Any]) -> Any:
    if isinstance(value, str):
        try:
            return value.format_map(context)
        except KeyError:
            return value
    if isinstance(value, list):
        return [_resolve_template(item, context) for item in value]
    if isinstance(value, dict):
        return {key: _resolve_template(item, context) for key, item in value.items()}
    return value


def _example_context(example: DatasetExample) -> Dict[str, Any]:
    context = example.to_dict()
    context.update({
        "id": example.example_id,
        "question": example.question,
        "prompt_prefix": example.prompt_prefix,
        "correct_answer": example.correct_answer or "",
        "wrong_answer": example.wrong_answer or "",
    })
    return context


def _assistant_start(stage1_prompt: str, assistant_prefix: str) -> int:
    assistant_start = stage1_prompt.lower().rfind("<think>")
    if assistant_start >= 0:
        return assistant_start
    if assistant_prefix and stage1_prompt.endswith(assistant_prefix):
        return len(stage1_prompt) - len(assistant_prefix)
    return len(stage1_prompt)


def _drop_closing_think(text: str) -> str:
    lower = text.lower()
    close_pos = lower.rfind("</think>")
    if close_pos < 0:
        return text
    return text[:close_pos]


def _prepare_stage1(
    backend,
    example: DatasetExample,
    generation_config: GenerationConfig,
    tracked_token_ids: List[int],
    use_prompt_prefix_first: bool,
) -> Dict[str, Any]:
    if use_prompt_prefix_first and example.prompt_prefix:
        assistant_stream = example.prompt_prefix
        return {
            "mode": "prompt_prefix",
            "stage1_prompt": None,
            "stage1_result": None,
            "assistant_stream": assistant_stream,
            "stage1_prefix_preamble": "",
        }

    if example.question:
        stage1_prompt = backend.build_prompt(example.question, generation_config)
        stage1_result = backend.generate(
            stage1_prompt,
            generation_config,
            stop_strings=[generation_config.stage1_stop_string] if generation_config.stage1_stop_string else None,
            tracked_token_ids=tracked_token_ids,
        )
        assistant_start = _assistant_start(stage1_prompt, generation_config.assistant_prefix)
        assistant_stream = stage1_result.full_text[assistant_start:]
        return {
            "mode": "question_stage1",
            "stage1_prompt": stage1_prompt,
            "stage1_result": stage1_result,
            "assistant_stream": assistant_stream,
            "stage1_prefix_preamble": stage1_result.full_text[:assistant_start],
        }

    if example.prompt_prefix:
        return {
            "mode": "prompt_prefix_fallback",
            "stage1_prompt": None,
            "stage1_result": None,
            "assistant_stream": example.prompt_prefix,
            "stage1_prefix_preamble": "",
        }

    raise ValueError(f"Example {example.example_id} has neither question nor prompt_prefix.")


def _prepare_stage1_batch(
    backend,
    examples: Sequence[DatasetExample],
    generation_config: GenerationConfig,
    tracked_token_ids: List[int],
    use_prompt_prefix_first: bool,
) -> List[Dict[str, Any]]:
    stage1_infos: List[Optional[Dict[str, Any]]] = [None] * len(examples)
    pending_indices: List[int] = []
    pending_prompts: List[str] = []

    for idx, example in enumerate(examples):
        has_prompt_prefix = bool(str(example.prompt_prefix or "").strip())
        has_question = bool(str(example.question or "").strip())
        if use_prompt_prefix_first and has_prompt_prefix:
            stage1_infos[idx] = {
                "mode": "prompt_prefix",
                "stage1_prompt": None,
                "stage1_result": None,
                "assistant_stream": example.prompt_prefix,
                "stage1_prefix_preamble": "",
            }
            continue
        if has_question:
            prompt = backend.build_prompt(example.question, generation_config)
            pending_indices.append(idx)
            pending_prompts.append(prompt)
            continue
        if has_prompt_prefix:
            stage1_infos[idx] = {
                "mode": "prompt_prefix_fallback",
                "stage1_prompt": None,
                "stage1_result": None,
                "assistant_stream": example.prompt_prefix,
                "stage1_prefix_preamble": "",
            }
            continue
        raise ValueError(f"Example {example.example_id} has neither question nor prompt_prefix.")

    if pending_prompts:
        stage1_results = backend.generate_many(
            pending_prompts,
            generation_config,
            stop_strings=[generation_config.stage1_stop_string] if generation_config.stage1_stop_string else None,
            tracked_token_ids=tracked_token_ids,
        )
        for idx, stage1_prompt, stage1_result in zip(pending_indices, pending_prompts, stage1_results):
            assistant_start = _assistant_start(stage1_prompt, generation_config.assistant_prefix)
            assistant_stream = stage1_result.full_text[assistant_start:]
            stage1_infos[idx] = {
                "mode": "question_stage1",
                "stage1_prompt": stage1_prompt,
                "stage1_result": stage1_result,
                "assistant_stream": assistant_stream,
                "stage1_prefix_preamble": stage1_result.full_text[:assistant_start],
            }

    resolved: List[Dict[str, Any]] = []
    for item in stage1_infos:
        if item is None:
            raise RuntimeError("Internal error: unresolved stage1 batch item.")
        resolved.append(item)
    return resolved


def _build_analysis_row(
    *,
    example: DatasetExample,
    stage1_info: Dict[str, Any],
    tampered_prefix: str,
    generation_result,
    edit_operations: List[Dict[str, Any]],
    condition_label: str,
    intervention_kind: str,
    keywords: Sequence[str],
    repair_lexicon: Sequence[str],
    tracked_strings: Sequence[str],
    backend,
    next_token_stats: Dict[str, Any],
    debug: Dict[str, Any],
) -> Dict[str, Any]:
    full_text = generation_result.full_text
    preamble = stage1_info.get("stage1_prefix_preamble", "") or ""
    assistant_full_text = full_text[len(preamble) :] if preamble and full_text.startswith(preamble) else full_text
    think_segments = extract_think_segments(assistant_full_text)
    think_text_full = "\n\n".join(think_segments)
    think_text_continuation = extract_continuation_think_text(generation_result.continuation)
    continuation_keyword_stats = analyze_text_keywords(think_text_continuation, keywords)
    full_keyword_stats = analyze_text_keywords(think_text_full, keywords)
    repair_signal_stats = analyze_repair_signals(think_text_continuation, repair_lexicon)
    final_boxed_answer = extract_last_boxed(generation_result.continuation)
    final_boxed_answer_full_text = extract_last_boxed(full_text)
    outcome = classify_outcome(final_boxed_answer, example.correct_answer, example.wrong_answer)
    token_target_stats = analyze_token_targets(
        backend=backend,
        full_text=assistant_full_text,
        think_text=think_text_continuation,
        generated_token_ids=generation_result.token_ids,
        tracked_strings=tracked_strings,
        step_scores=generation_result.step_scores,
    )
    return {
        "example_id": example.example_id,
        "question": example.question,
        "prompt_prefix": example.prompt_prefix,
        "correct_answer": example.correct_answer,
        "wrong_answer": example.wrong_answer,
        "metadata": example.metadata,
        "stage_mode": stage1_info["mode"],
        "stage1_prompt": stage1_info.get("stage1_prompt"),
        "stage1_full_text": stage1_info["stage1_result"].full_text if stage1_info.get("stage1_result") else None,
        "stage1_generated_tokens": stage1_info["stage1_result"].generated_tokens if stage1_info.get("stage1_result") else None,
        "assistant_stream_before_edit": stage1_info.get("assistant_stream"),
        "edit_operations": edit_operations,
        "tampered_prefix": tampered_prefix,
        "assistant_full_text": assistant_full_text,
        "condition_label": condition_label,
        "intervention_kind": intervention_kind,
        "generated_continuation": generation_result.continuation,
        "full_text": full_text,
        "think_text_full": think_text_full,
        "think_text_continuation": think_text_continuation,
        "final_boxed_answer": final_boxed_answer,
        "final_boxed_answer_full_text": final_boxed_answer_full_text,
        "outcome": outcome,
        "generated_tokens": generation_result.generated_tokens,
        "continuation_keyword_stats": continuation_keyword_stats,
        "full_keyword_stats": full_keyword_stats,
        "repair_signal_stats": repair_signal_stats,
        "token_target_stats": token_target_stats,
        "next_token_stats": next_token_stats,
        "debug": debug,
    }


def _resolve_next_token_candidates(
    backend,
    candidate_texts: Sequence[str],
) -> Tuple[List[int], Dict[str, str], List[Dict[str, Any]]]:
    token_ids: List[int] = []
    labels: Dict[str, str] = {}
    skipped: List[Dict[str, Any]] = []
    for text in candidate_texts:
        encoded = backend.encode(text)
        if len(encoded) != 1:
            skipped.append({"text": text, "token_ids": encoded})
            continue
        token_id = int(encoded[0])
        if token_id not in token_ids:
            token_ids.append(token_id)
        labels[str(token_id)] = text
    return token_ids, labels, skipped


def run_experiment(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    dataset_cfg = dict(config.get("dataset", {}))
    output_dir = Path(args.output_dir or config.get("output_dir") or "outputs/cot_research")
    output_dir.mkdir(parents=True, exist_ok=True)

    generation_config = _make_generation_config(config)
    backend_config = _make_backend_config(config, args)
    backend = create_backend(backend_config)

    examples = load_examples(str(dataset_cfg["path"]))
    if args.max_examples > 0:
        examples = examples[: args.max_examples]
    elif int(dataset_cfg.get("max_examples", -1)) > 0:
        examples = examples[: int(dataset_cfg.get("max_examples", -1))]

    analysis_cfg = dict(config.get("analysis", {}))
    keywords = list(analysis_cfg.get("keywords") or DEFAULT_KEYWORDS)
    repair_lexicon = list(analysis_cfg.get("repair_lexicon") or DEFAULT_REPAIR_LEXICON)
    tracked_strings = list(analysis_cfg.get("tracked_strings") or [])
    next_token_candidate_texts = list(analysis_cfg.get("next_token_candidates") or tracked_strings)
    next_token_ids, next_token_labels, skipped_next_token_candidates = _resolve_next_token_candidates(
        backend,
        next_token_candidate_texts,
    )

    editing_plan = list(config.get("editing_plan") or [])
    interventions = list(config.get("interventions") or [])
    use_prompt_prefix_first = bool(dataset_cfg.get("use_prompt_prefix_first", False))

    all_rows: List[Dict[str, Any]] = []
    model = getattr(backend, "model", None)
    attn_modules = None
    if interventions:
        if not backend.supports_intervention or model is None:
            raise ValueError("Configured interventions require an HF backend with a loaded model.")
        _, attn_modules, _ = resolve_head_targets(model, [])

    for example in examples:
        context = _example_context(example)
        stage1_info = _prepare_stage1(
            backend=backend,
            example=example,
            generation_config=generation_config,
            tracked_token_ids=next_token_ids,
            use_prompt_prefix_first=use_prompt_prefix_first,
        )
        resolved_plan = _resolve_template(editing_plan, context)
        edited_stream, edit_ops = apply_editor_chain(stage1_info["assistant_stream"], resolved_plan)
        tampered_prefix = stage1_info["stage1_prefix_preamble"] + _drop_closing_think(edited_stream)

        baseline_generation = backend.generate(
            tampered_prefix,
            generation_config,
            tracked_token_ids=next_token_ids,
        )
        baseline_next_stats = backend.next_token_stats(tampered_prefix, next_token_ids) if next_token_ids else {}
        baseline_row = _build_analysis_row(
            example=example,
            stage1_info=stage1_info,
            tampered_prefix=tampered_prefix,
            generation_result=baseline_generation,
            edit_operations=[item.to_dict() for item in edit_ops],
            condition_label="baseline",
            intervention_kind="baseline",
            keywords=keywords,
            repair_lexicon=repair_lexicon,
            tracked_strings=tracked_strings,
            backend=backend,
            next_token_stats={"candidate_stats": baseline_next_stats, "candidate_labels": next_token_labels},
            debug={},
        )
        all_rows.append(baseline_row)

        for item in interventions:
            resolved_item = _resolve_template(item, context)
            intervention_name = str(resolved_item["name"])
            heads = list(resolved_item.get("heads") or [])
            if not heads:
                raise ValueError(f"Intervention '{intervention_name}' requires non-empty heads.")
            targets, attn_modules, _ = resolve_head_targets(model, heads)
            operations = INTERVENTION_REGISTRY.get_required(intervention_name)(targets, dict(resolved_item.get("params", {})))
            label = str(resolved_item.get("label") or f"{intervention_name}[{','.join(head.label for head in targets)}]")
            with MultiLayerHeadIntervention(attn_modules, operations) as hook_set:
                intervention_next_stats = backend.next_token_stats(tampered_prefix, next_token_ids) if next_token_ids else {}
                intervention_generation = backend.generate(
                    tampered_prefix,
                    generation_config,
                    tracked_token_ids=next_token_ids,
                )
            row = _build_analysis_row(
                example=example,
                stage1_info=stage1_info,
                tampered_prefix=tampered_prefix,
                generation_result=intervention_generation,
                edit_operations=[op.to_dict() for op in edit_ops],
                condition_label=label,
                intervention_kind=intervention_name,
                keywords=keywords,
                repair_lexicon=repair_lexicon,
                tracked_strings=tracked_strings,
                backend=backend,
                next_token_stats={"candidate_stats": intervention_next_stats, "candidate_labels": next_token_labels},
                debug=hook_set.merged_debug_state(),
            )
            all_rows.append(row)

    rows_path = output_dir / "rows.jsonl"
    summary_path = output_dir / "summary.csv"
    token_summary_path = output_dir / "token_target_summary.csv"
    next_token_summary_path = output_dir / "next_token_summary.csv"
    summary_json_path = output_dir / "summary.json"
    run_config_path = output_dir / "run_config.json"

    dump_jsonl(rows_path, all_rows)
    condition_summary = summarize_condition_rows(all_rows)
    token_summary = summarize_token_targets(all_rows)
    next_token_summary = summarize_next_token_targets(all_rows)

    write_csv(summary_path, condition_summary)
    write_csv(token_summary_path, token_summary)
    write_csv(next_token_summary_path, next_token_summary)
    write_json(
        summary_json_path,
        {
            "condition_summary": condition_summary,
            "token_target_summary": token_summary,
            "next_token_summary": next_token_summary,
        },
    )
    write_json(
        run_config_path,
        {
            "config": config,
            "resolved_backend": backend_config.__dict__,
            "resolved_generation": generation_config.__dict__,
            "example_count": len(examples),
            "analysis": {
                "keywords": keywords,
                "repair_lexicon": repair_lexicon,
                "tracked_strings": tracked_strings,
                "next_token_candidate_texts": next_token_candidate_texts,
                "skipped_next_token_candidates": skipped_next_token_candidates,
            },
        },
    )

    return {
        "output_dir": str(output_dir),
        "rows_path": str(rows_path),
        "summary_path": str(summary_path),
        "token_summary_path": str(token_summary_path),
        "next_token_summary_path": str(next_token_summary_path),
        "summary_json_path": str(summary_json_path),
        "run_config_path": str(run_config_path),
        "row_count": len(all_rows),
    }


def main() -> None:
    args = parse_args()
    config = _load_config(args.config)
    result = run_experiment(config, args)
    print("[Done] CoT experiment outputs written:")
    print(f"- output_dir: {result['output_dir']}")
    print(f"- rows: {result['rows_path']}")
    print(f"- summary: {result['summary_path']}")
    print(f"- token_summary: {result['token_summary_path']}")
    print(f"- next_token_summary: {result['next_token_summary_path']}")
    print(f"- summary_json: {result['summary_json_path']}")
    print(f"- run_config: {result['run_config_path']}")
    print(f"- row_count: {result['row_count']}")


if __name__ == "__main__":
    main()
