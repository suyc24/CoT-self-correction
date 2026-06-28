from .answer_extraction import answers_match, classify_outcome, extract_last_boxed
from .attention_sink_analysis import (
    SinkConfig,
    aggregate_head_metrics,
    annotate_example_head_ranks,
    compute_head_sink_metrics,
    resolve_sink_positions,
)
from .cot_accuracy import (
    build_row_lookup,
    classify_answer_change,
    extract_final_answer_from_row,
    judge_comparison_row,
    judge_single_row,
    resolve_gold_answer,
)
from .cot_editing import EDITOR_REGISTRY, apply_editor_chain
from .datasets import convert_numinamath_row, load_examples, load_numinamath_dataset
from .generation import create_backend, generate_from_prefix, generate_full_cot, generate_stage1_until_first_think_close
from .head_ablation import HeadSpec, MultiHeadAblationHookSet, SingleHeadAblationHook, filter_heads, list_all_heads
from .head_intervention import INTERVENTION_REGISTRY, list_model_heads
from .io_utils import dump_jsonl, load_jsonl, peek_first_json_row, truncate_text, write_csv, write_json
from .local_attention_analysis import (
    aggregate_query_metrics,
    annotate_query_metric_ranks,
    compute_query_attention_metrics,
    locate_close_think_query,
    parse_int_list,
)
from .model_utils import load_model_with_retries, resolve_target_token_id
from .prompt_utils import build_chat_prompt
from .repetition_analysis import RepetitionThresholds, analyze_repetition, analyze_row_repetition
from .row_utils import (
    build_example_id,
    build_problem_text,
    build_reference_solution,
    resolve_prompt_prefix_from_row,
    resolve_row_identities,
    select_continuation_text,
)
from .runtime_utils import parse_parallel_gpu_ids, resolve_device_map, seed_everything, split_examples_contiguous
from .self_correction import (
    ABLATION_POSITION,
    DEFAULT_KEYWORDS,
    analyze_generation,
    parse_keywords,
    prepare_example_prefix,
)
from .text_analysis import analyze_repair_signals, analyze_text_keywords

__all__ = [
    "ABLATION_POSITION",
    "DEFAULT_KEYWORDS",
    "EDITOR_REGISTRY",
    "HeadSpec",
    "INTERVENTION_REGISTRY",
    "MultiHeadAblationHookSet",
    "RepetitionThresholds",
    "SingleHeadAblationHook",
    "SinkConfig",
    "analyze_repetition",
    "analyze_repair_signals",
    "analyze_row_repetition",
    "analyze_generation",
    "analyze_text_keywords",
    "aggregate_head_metrics",
    "aggregate_query_metrics",
    "annotate_example_head_ranks",
    "annotate_query_metric_ranks",
    "answers_match",
    "apply_editor_chain",
    "build_row_lookup",
    "build_chat_prompt",
    "build_example_id",
    "build_problem_text",
    "build_reference_solution",
    "classify_outcome",
    "classify_answer_change",
    "compute_head_sink_metrics",
    "compute_query_attention_metrics",
    "convert_numinamath_row",
    "create_backend",
    "dump_jsonl",
    "extract_final_answer_from_row",
    "extract_last_boxed",
    "generate_from_prefix",
    "generate_full_cot",
    "generate_stage1_until_first_think_close",
    "filter_heads",
    "judge_comparison_row",
    "judge_single_row",
    "load_examples",
    "load_jsonl",
    "load_model_with_retries",
    "load_numinamath_dataset",
    "list_all_heads",
    "list_model_heads",
    "locate_close_think_query",
    "parse_int_list",
    "parse_keywords",
    "parse_parallel_gpu_ids",
    "peek_first_json_row",
    "prepare_example_prefix",
    "resolve_sink_positions",
    "resolve_device_map",
    "resolve_gold_answer",
    "resolve_prompt_prefix_from_row",
    "resolve_target_token_id",
    "resolve_row_identities",
    "seed_everything",
    "select_continuation_text",
    "split_examples_contiguous",
    "truncate_text",
    "write_csv",
    "write_json",
]
