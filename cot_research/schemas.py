from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DatasetExample:
    example_id: str
    question: str = ""
    prompt_prefix: str = ""
    correct_answer: Optional[str] = None
    wrong_answer: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, obj: Dict[str, Any]) -> "DatasetExample":
        if "id" not in obj:
            raise ValueError("Dataset record missing required field 'id'.")
        metadata = obj.get("metadata") or {}
        if not isinstance(metadata, dict):
            raise ValueError("Dataset record field 'metadata' must be an object when provided.")
        return cls(
            example_id=str(obj["id"]),
            question=str(obj.get("question", "") or ""),
            prompt_prefix=str(obj.get("prompt_prefix", "") or ""),
            correct_answer=None if obj.get("correct_answer") is None else str(obj.get("correct_answer")),
            wrong_answer=None if obj.get("wrong_answer") is None else str(obj.get("wrong_answer")),
            metadata=metadata,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.example_id,
            "question": self.question,
            "prompt_prefix": self.prompt_prefix,
            "correct_answer": self.correct_answer,
            "wrong_answer": self.wrong_answer,
            "metadata": self.metadata,
        }


@dataclass
class GenerationConfig:
    system_prompt: str
    assistant_prefix: str = "<think>\n"
    stage1_stop_string: str = "</think>"
    max_stage1_tokens: int = 2048
    max_new_tokens: int = 1024
    do_sample: bool = False
    temperature: float = 0.7
    top_k: int = 0
    top_p: float = 0.9
    repetition_penalty: float = 1.0
    enable_thinking: bool = True
    capture_step_scores: bool = False


@dataclass
class BackendConfig:
    backend_type: str = "auto"
    model_name_or_path: str = "Qwen/Qwen3-4B"
    device_map: Any = "auto"
    load_in_half: bool = True
    use_fast_tokenizer: bool = False
    use_safetensors: bool = True
    local_files_only: bool = True
    attn_implementation: str = ""
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    max_num_seqs: int = 256
    max_num_batched_tokens: Optional[int] = None
    enable_chunked_prefill: bool = True
    enforce_eager: bool = False
    disable_log_stats: bool = True
    mock_responses: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KeywordHit:
    keyword: str
    start: int
    end: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TokenCountResult:
    target: str
    text_count_full: int = 0
    text_count_think: int = 0
    first_text_pos_full: Optional[int] = None
    first_text_pos_think: Optional[int] = None
    token_count_full: int = 0
    token_count_think: int = 0
    first_token_index_full: Optional[int] = None
    first_token_index_think: Optional[int] = None
    tracked_token_ids: List[int] = field(default_factory=list)
    step_scores: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EditOperation:
    editor_name: str
    success: bool
    input_text: str
    output_text: str
    absolute_start: Optional[int] = None
    absolute_end: Optional[int] = None
    replacement_text: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GenerationResult:
    prompt_prefix: str
    continuation: str
    full_text: str
    generated_tokens: int
    token_ids: List[int] = field(default_factory=list)
    step_scores: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class InterventionResult:
    label: str
    kind: str
    debug: Dict[str, Any] = field(default_factory=dict)
    next_token_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
