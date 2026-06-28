from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence

import torch
from transformers import AutoTokenizer, StoppingCriteriaList

from .model_utils import (
    StopOnTokenSequences,
    get_input_device_for_model,
    load_hf_model_and_tokenizer,
    prepare_inputs_for_model,
)
from .prompt_utils import build_chat_prompt
from .schemas import BackendConfig, GenerationConfig, GenerationResult


class GenerationBackend(ABC):
    @abstractmethod
    def build_prompt(self, question: str, generation_config: GenerationConfig) -> str:
        raise NotImplementedError

    @abstractmethod
    def encode(self, text: str) -> List[int]:
        raise NotImplementedError

    @abstractmethod
    def decode(self, token_ids: Sequence[int]) -> str:
        raise NotImplementedError

    @abstractmethod
    def generate(
        self,
        prompt_prefix: str,
        generation_config: GenerationConfig,
        stop_strings: Optional[List[str]] = None,
        tracked_token_ids: Optional[List[int]] = None,
    ) -> GenerationResult:
        raise NotImplementedError

    def generate_many(
        self,
        prompt_prefixes: Sequence[str],
        generation_config: GenerationConfig,
        stop_strings: Optional[List[str]] = None,
        tracked_token_ids: Optional[List[int]] = None,
    ) -> List[GenerationResult]:
        return [
            self.generate(
                prompt_prefix,
                generation_config,
                stop_strings=stop_strings,
                tracked_token_ids=tracked_token_ids,
            )
            for prompt_prefix in prompt_prefixes
        ]

    def generate_chat(
        self,
        messages: Sequence[Dict[str, str]],
        generation_config: GenerationConfig,
        stop_strings: Optional[List[str]] = None,
        tracked_token_ids: Optional[List[int]] = None,
    ) -> GenerationResult:
        prompt_prefix = self.build_prompt(str(messages[-1].get("content") or ""), generation_config)
        return self.generate(
            prompt_prefix,
            generation_config,
            stop_strings=stop_strings,
            tracked_token_ids=tracked_token_ids,
        )

    def generate_many_chat(
        self,
        message_batches: Sequence[Sequence[Dict[str, str]]],
        generation_config: GenerationConfig,
        stop_strings: Optional[List[str]] = None,
        tracked_token_ids: Optional[List[int]] = None,
    ) -> List[GenerationResult]:
        return [
            self.generate_chat(
                messages,
                generation_config,
                stop_strings=stop_strings,
                tracked_token_ids=tracked_token_ids,
            )
            for messages in message_batches
        ]

    @abstractmethod
    def next_token_stats(self, prompt_prefix: str, token_ids: Sequence[int]) -> Dict[str, Dict[str, float]]:
        raise NotImplementedError

    @property
    @abstractmethod
    def supports_intervention(self) -> bool:
        raise NotImplementedError

    @property
    def supports_next_token_stats(self) -> bool:
        return True

    @property
    def supports_batch_generation(self) -> bool:
        return False

    @property
    def model(self) -> Optional[torch.nn.Module]:
        return None

    @property
    def tokenizer(self):
        return None


class MockGenerationBackend(GenerationBackend):
    def __init__(self, config: BackendConfig) -> None:
        self.config = config

    @property
    def supports_intervention(self) -> bool:
        return False

    def build_prompt(self, question: str, generation_config: GenerationConfig) -> str:
        return f"System: {generation_config.system_prompt}\nUser: {question}\nAssistant:\n{generation_config.assistant_prefix}"

    def encode(self, text: str) -> List[int]:
        return [ord(ch) for ch in text]

    def decode(self, token_ids: Sequence[int]) -> str:
        return "".join(chr(token_id) for token_id in token_ids)

    def _select_response(self, prompt_prefix: str, stop_strings: Optional[List[str]]) -> str:
        for key, value in sorted(self.config.mock_responses.items(), key=lambda item: len(item[0]), reverse=True):
            if prompt_prefix.endswith(key) or key in prompt_prefix:
                if isinstance(value, dict):
                    if stop_strings:
                        return str(value.get("stage1", ""))
                    return str(value.get("continuation", value.get("default", "")))
                return str(value)
        if stop_strings:
            return "mock reasoning </think>"
        return "mock continuation \\boxed{0}"

    def generate(
        self,
        prompt_prefix: str,
        generation_config: GenerationConfig,
        stop_strings: Optional[List[str]] = None,
        tracked_token_ids: Optional[List[int]] = None,
    ) -> GenerationResult:
        continuation = self._select_response(prompt_prefix, stop_strings)
        token_ids = self.encode(continuation)
        step_scores: List[Dict[str, Any]] = []
        if tracked_token_ids:
            for step, token_id in enumerate(token_ids):
                step_scores.append(
                    {
                        "step": step,
                        "generated_token_id": token_id,
                        "generated_text": self.decode([token_id]),
                        "candidates": {
                            str(candidate): {
                                "logit": 0.0 if candidate == token_id else -5.0,
                                "prob": 1.0 if candidate == token_id else 0.0,
                            }
                            for candidate in tracked_token_ids
                        },
                    }
                )
        return GenerationResult(
            prompt_prefix=prompt_prefix,
            continuation=continuation,
            full_text=prompt_prefix + continuation,
            generated_tokens=len(token_ids),
            token_ids=token_ids,
            step_scores=step_scores,
        )

    def next_token_stats(self, prompt_prefix: str, token_ids: Sequence[int]) -> Dict[str, Dict[str, float]]:
        return {str(token_id): {"logit": 0.0, "prob": 1.0 / max(len(token_ids), 1)} for token_id in token_ids}


class _TokenizerMixin:
    def _ensure_tokenizer_special_tokens(self) -> None:
        tokenizer = self._tokenizer
        if tokenizer.pad_token is None:
            if tokenizer.unk_token:
                tokenizer.pad_token = tokenizer.unk_token
                tokenizer.pad_token_id = tokenizer.unk_token_id
            elif tokenizer.eos_token:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id
            else:
                raise ValueError("Tokenizer has no pad/unk/eos token.")

    @property
    def tokenizer(self):
        return self._tokenizer

    def build_prompt(self, question: str, generation_config: GenerationConfig) -> str:
        return build_chat_prompt(
            self._tokenizer,
            question=question,
            system_prompt=generation_config.system_prompt,
            assistant_prefix=generation_config.assistant_prefix,
            enable_thinking=generation_config.enable_thinking,
        )

    def encode(self, text: str) -> List[int]:
        return self._tokenizer.encode(text, add_special_tokens=False)

    def decode(self, token_ids: Sequence[int]) -> str:
        return self._tokenizer.decode(list(token_ids), skip_special_tokens=False)


class HFGenerationBackend(_TokenizerMixin, GenerationBackend):
    def __init__(self, config: BackendConfig) -> None:
        self.config = config
        self._model, self._tokenizer = load_hf_model_and_tokenizer(
            model_name_or_path=config.model_name_or_path,
            load_in_half=config.load_in_half,
            use_fast_tokenizer=config.use_fast_tokenizer,
            use_safetensors=config.use_safetensors,
            local_files_only=config.local_files_only,
            device_map=config.device_map,
            attn_implementation=config.attn_implementation,
        )
        self._ensure_tokenizer_special_tokens()

    @property
    def supports_intervention(self) -> bool:
        return True

    @property
    def supports_batch_generation(self) -> bool:
        return True

    @property
    def model(self) -> Optional[torch.nn.Module]:
        return self._model

    def _input_device(self) -> torch.device:
        return get_input_device_for_model(self._model)

    def _prepare_inputs(self, text: str) -> Dict[str, torch.Tensor]:
        return prepare_inputs_for_model(self._tokenizer, text, self._model)

    def _prepare_batched_inputs(self, texts: Sequence[str]) -> Dict[str, torch.Tensor]:
        inputs = self._tokenizer(
            list(texts),
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        model_device = self._input_device()
        inputs["input_ids"] = inputs["input_ids"].to(model_device)
        if "attention_mask" in inputs:
            inputs["attention_mask"] = inputs["attention_mask"].to(model_device)
        return inputs

    def generate(
        self,
        prompt_prefix: str,
        generation_config: GenerationConfig,
        stop_strings: Optional[List[str]] = None,
        tracked_token_ids: Optional[List[int]] = None,
    ) -> GenerationResult:
        inputs = self._prepare_inputs(prompt_prefix)
        generation_kwargs: Dict[str, Any] = {
            "input_ids": inputs["input_ids"],
            "max_new_tokens": generation_config.max_stage1_tokens if stop_strings else generation_config.max_new_tokens,
            "do_sample": generation_config.do_sample,
            "pad_token_id": self._tokenizer.pad_token_id,
            "return_dict_in_generate": True,
            "output_scores": bool(generation_config.capture_step_scores and tracked_token_ids),
        }
        if "attention_mask" in inputs:
            generation_kwargs["attention_mask"] = inputs["attention_mask"]
        if self._tokenizer.eos_token_id is not None:
            generation_kwargs["eos_token_id"] = self._tokenizer.eos_token_id
        if generation_config.do_sample:
            generation_kwargs["temperature"] = generation_config.temperature
            if int(generation_config.top_k or 0) > 0:
                generation_kwargs["top_k"] = int(generation_config.top_k)
            generation_kwargs["top_p"] = generation_config.top_p
        if float(generation_config.repetition_penalty or 1.0) != 1.0:
            generation_kwargs["repetition_penalty"] = float(generation_config.repetition_penalty)
        if stop_strings:
            sequences = [self.encode(item) for item in stop_strings if item]
            sequences = [item for item in sequences if item]
            if sequences:
                generation_kwargs["stopping_criteria"] = StoppingCriteriaList([StopOnTokenSequences(sequences)])
        with torch.no_grad():
            outputs = self._model.generate(**generation_kwargs)
        sequences = outputs.sequences[0]
        prompt_len = int(inputs["input_ids"].shape[-1])
        new_token_ids = sequences[prompt_len:]
        continuation = self._tokenizer.decode(new_token_ids, skip_special_tokens=True)
        step_scores: List[Dict[str, Any]] = []
        if tracked_token_ids and getattr(outputs, "scores", None) is not None:
            for step, score_tensor in enumerate(outputs.scores):
                logits = score_tensor[0]
                probs = torch.softmax(logits, dim=-1)
                generated_token_id = int(new_token_ids[step].item()) if step < len(new_token_ids) else None
                candidates = {}
                for token_id in tracked_token_ids:
                    candidates[str(token_id)] = {
                        "logit": float(logits[token_id].detach().item()),
                        "prob": float(probs[token_id].detach().item()),
                    }
                step_scores.append(
                    {
                        "step": step,
                        "generated_token_id": generated_token_id,
                        "generated_text": self.decode([generated_token_id]) if generated_token_id is not None else "",
                        "candidates": candidates,
                    }
                )
        return GenerationResult(
            prompt_prefix=prompt_prefix,
            continuation=continuation,
            full_text=prompt_prefix + continuation,
            generated_tokens=int(new_token_ids.shape[0]),
            token_ids=[int(token_id) for token_id in new_token_ids.tolist()],
            step_scores=step_scores,
        )

    def generate_many(
        self,
        prompt_prefixes: Sequence[str],
        generation_config: GenerationConfig,
        stop_strings: Optional[List[str]] = None,
        tracked_token_ids: Optional[List[int]] = None,
    ) -> List[GenerationResult]:
        if not prompt_prefixes:
            return []

        # Batched stopping criteria can terminate the whole batch when any one
        # sample matches a stop sequence, so fall back to per-example generation.
        if stop_strings or (generation_config.capture_step_scores and tracked_token_ids):
            return super().generate_many(
                prompt_prefixes,
                generation_config,
                stop_strings=stop_strings,
                tracked_token_ids=tracked_token_ids,
            )

        inputs = self._prepare_batched_inputs(prompt_prefixes)
        generation_kwargs: Dict[str, Any] = {
            "input_ids": inputs["input_ids"],
            "max_new_tokens": generation_config.max_new_tokens,
            "do_sample": generation_config.do_sample,
            "pad_token_id": self._tokenizer.pad_token_id,
            "return_dict_in_generate": True,
            "output_scores": False,
        }
        if "attention_mask" in inputs:
            generation_kwargs["attention_mask"] = inputs["attention_mask"]
        if self._tokenizer.eos_token_id is not None:
            generation_kwargs["eos_token_id"] = self._tokenizer.eos_token_id
        if generation_config.do_sample:
            generation_kwargs["temperature"] = generation_config.temperature
            if int(generation_config.top_k or 0) > 0:
                generation_kwargs["top_k"] = int(generation_config.top_k)
            generation_kwargs["top_p"] = generation_config.top_p
        if float(generation_config.repetition_penalty or 1.0) != 1.0:
            generation_kwargs["repetition_penalty"] = float(generation_config.repetition_penalty)

        with torch.no_grad():
            outputs = self._model.generate(**generation_kwargs)

        sequences = outputs.sequences
        prompt_len = int(inputs["input_ids"].shape[-1])
        results: List[GenerationResult] = []
        for row_idx, prompt_prefix in enumerate(prompt_prefixes):
            new_token_ids = sequences[row_idx, prompt_len:]
            continuation = self._tokenizer.decode(new_token_ids, skip_special_tokens=True)
            results.append(
                GenerationResult(
                    prompt_prefix=prompt_prefix,
                    continuation=continuation,
                    full_text=prompt_prefix + continuation,
                    generated_tokens=int(new_token_ids.shape[0]),
                    token_ids=[int(token_id) for token_id in new_token_ids.tolist()],
                    step_scores=[],
                )
            )
        return results

    def next_token_stats(self, prompt_prefix: str, token_ids: Sequence[int]) -> Dict[str, Dict[str, float]]:
        inputs = self._prepare_inputs(prompt_prefix)
        with torch.no_grad():
            outputs = self._model(**inputs)
        logits = outputs.logits[0, -1]
        probs = torch.softmax(logits, dim=-1)
        return {
            str(token_id): {
                "logit": float(logits[token_id].detach().item()),
                "prob": float(probs[token_id].detach().item()),
            }
            for token_id in token_ids
        }


class VLLMGenerationBackend(_TokenizerMixin, GenerationBackend):
    def __init__(self, config: BackendConfig) -> None:
        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise RuntimeError(
                "This backend requires the `vllm` package. Install it into the existing environment, for example:\n"
                "  pip install vllm\n"
                "No new environment is required."
            ) from exc
        if not torch.cuda.is_available():
            raise RuntimeError("vLLM backend requires CUDA, but torch.cuda.is_available() is False.")
        self.config = config
        self._SamplingParams = SamplingParams
        self._tokenizer = AutoTokenizer.from_pretrained(
            config.model_name_or_path,
            use_fast=config.use_fast_tokenizer,
            padding_side="left",
            trust_remote_code=True,
            local_files_only=config.local_files_only,
        )
        self._ensure_tokenizer_special_tokens()
        llm_kwargs: Dict[str, Any] = {
            "model": config.model_name_or_path,
            "tokenizer": config.model_name_or_path,
            "trust_remote_code": True,
            "tensor_parallel_size": max(int(config.tensor_parallel_size or 1), 1),
            "gpu_memory_utilization": float(config.gpu_memory_utilization),
            "max_num_seqs": int(config.max_num_seqs),
            "enable_chunked_prefill": bool(config.enable_chunked_prefill),
            "enforce_eager": bool(config.enforce_eager),
            "disable_log_stats": bool(config.disable_log_stats),
            "dtype": "half" if config.load_in_half else "auto",
        }
        if config.max_num_batched_tokens is not None:
            llm_kwargs["max_num_batched_tokens"] = int(config.max_num_batched_tokens)
        self._llm = LLM(**llm_kwargs)

    @property
    def supports_intervention(self) -> bool:
        return False

    @property
    def supports_next_token_stats(self) -> bool:
        return False

    @property
    def supports_batch_generation(self) -> bool:
        return True

    def _build_sampling_params(
        self,
        generation_config: GenerationConfig,
        stop_strings: Optional[List[str]],
        *,
        max_tokens: Optional[int] = None,
        logprobs: Optional[int] = None,
    ):
        max_tokens = generation_config.max_stage1_tokens if (stop_strings and max_tokens is None) else max_tokens
        if max_tokens is None:
            max_tokens = generation_config.max_new_tokens
        kwargs: Dict[str, Any] = {
            "n": 1,
            "max_tokens": int(max_tokens),
            "temperature": float(generation_config.temperature) if generation_config.do_sample else 0.0,
            "top_k": int(generation_config.top_k) if generation_config.do_sample and int(generation_config.top_k or 0) > 0 else -1,
            "top_p": float(generation_config.top_p) if generation_config.do_sample else 1.0,
            "repetition_penalty": float(generation_config.repetition_penalty or 1.0),
        }
        if stop_strings:
            kwargs["stop"] = list(stop_strings)
        if logprobs is not None:
            kwargs["logprobs"] = int(logprobs)
        try:
            sig = inspect.signature(self._SamplingParams.__init__)
            supported = set(sig.parameters.keys())
        except (TypeError, ValueError):
            supported = set(kwargs.keys())
        if "include_stop_str_in_output" in supported:
            kwargs["include_stop_str_in_output"] = True
        if "skip_special_tokens" in supported:
            kwargs["skip_special_tokens"] = False
        if "detokenize" in supported:
            kwargs["detokenize"] = True
        if supported == {"self", "args", "kwargs"}:
            return self._SamplingParams(**kwargs)
        filtered = {key: value for key, value in kwargs.items() if key in supported}
        return self._SamplingParams(**filtered)

    def _result_from_vllm_output(self, prompt_prefix: str, output: Any) -> GenerationResult:
        result = output.outputs[0]
        token_ids = [int(token_id) for token_id in list(getattr(result, "token_ids", []) or [])]
        continuation = str(getattr(result, "text", ""))
        return GenerationResult(
            prompt_prefix=prompt_prefix,
            continuation=continuation,
            full_text=prompt_prefix + continuation,
            generated_tokens=len(token_ids),
            token_ids=token_ids,
            step_scores=[],
        )

    def _continue_final_message(self, generation_config: GenerationConfig) -> bool:
        return bool(generation_config.assistant_prefix)

    def _prepare_chat_messages(
        self,
        messages: Sequence[Dict[str, str]],
        generation_config: GenerationConfig,
    ) -> List[Dict[str, str]]:
        prepared = [dict(item) for item in messages]
        if generation_config.assistant_prefix:
            prepared.append({"role": "assistant", "content": str(generation_config.assistant_prefix)})
        return prepared

    def generate(
        self,
        prompt_prefix: str,
        generation_config: GenerationConfig,
        stop_strings: Optional[List[str]] = None,
        tracked_token_ids: Optional[List[int]] = None,
    ) -> GenerationResult:
        sampling_params = self._build_sampling_params(generation_config, stop_strings)
        outputs = self._llm.generate([prompt_prefix], sampling_params, use_tqdm=False)
        return self._result_from_vllm_output(prompt_prefix, outputs[0])

    def generate_chat(
        self,
        messages: Sequence[Dict[str, str]],
        generation_config: GenerationConfig,
        stop_strings: Optional[List[str]] = None,
        tracked_token_ids: Optional[List[int]] = None,
    ) -> GenerationResult:
        sampling_params = self._build_sampling_params(generation_config, stop_strings)
        prepared_messages = self._prepare_chat_messages(messages, generation_config)
        outputs = self._llm.chat(
            [prepared_messages],
            sampling_params=sampling_params,
            use_tqdm=False,
            add_generation_prompt=not self._continue_final_message(generation_config),
            continue_final_message=self._continue_final_message(generation_config),
            chat_template_kwargs={"enable_thinking": bool(generation_config.enable_thinking)},
        )
        return self._result_from_vllm_output("", outputs[0])

    def generate_many(
        self,
        prompt_prefixes: Sequence[str],
        generation_config: GenerationConfig,
        stop_strings: Optional[List[str]] = None,
        tracked_token_ids: Optional[List[int]] = None,
    ) -> List[GenerationResult]:
        if not prompt_prefixes:
            return []
        sampling_params = self._build_sampling_params(generation_config, stop_strings)
        outputs = self._llm.generate(list(prompt_prefixes), sampling_params, use_tqdm=False)
        return [self._result_from_vllm_output(prompt_prefix, output) for prompt_prefix, output in zip(prompt_prefixes, outputs)]

    def generate_many_chat(
        self,
        message_batches: Sequence[Sequence[Dict[str, str]]],
        generation_config: GenerationConfig,
        stop_strings: Optional[List[str]] = None,
        tracked_token_ids: Optional[List[int]] = None,
    ) -> List[GenerationResult]:
        if not message_batches:
            return []
        sampling_params = self._build_sampling_params(generation_config, stop_strings)
        prepared_batches = [
            self._prepare_chat_messages(messages, generation_config)
            for messages in message_batches
        ]
        outputs = self._llm.chat(
            prepared_batches,
            sampling_params=sampling_params,
            use_tqdm=False,
            add_generation_prompt=not self._continue_final_message(generation_config),
            continue_final_message=self._continue_final_message(generation_config),
            chat_template_kwargs={"enable_thinking": bool(generation_config.enable_thinking)},
        )
        return [self._result_from_vllm_output("", output) for output in outputs]

    def next_token_stats(self, prompt_prefix: str, token_ids: Sequence[int]) -> Dict[str, Dict[str, float]]:
        return {}


def create_backend(config: BackendConfig) -> GenerationBackend:
    backend_type = config.backend_type.strip().lower()
    if backend_type == "mock":
        return MockGenerationBackend(config)
    if backend_type == "hf":
        return HFGenerationBackend(config)
    if backend_type == "vllm":
        return VLLMGenerationBackend(config)
    if backend_type == "auto":
        try:
            if torch.cuda.is_available():
                return VLLMGenerationBackend(config)
        except Exception:
            pass
        return HFGenerationBackend(config)
    raise ValueError(f"Unsupported backend type: {config.backend_type}")


def _sample_next_token_id(
    logits: torch.Tensor,
    *,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> int:
    if not do_sample:
        return int(torch.argmax(logits).item())

    scaled_logits = logits
    if temperature > 0:
        scaled_logits = logits / max(float(temperature), 1e-5)
    probs = torch.softmax(scaled_logits, dim=-1)

    if 0.0 < top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        keep_mask = (cumulative_probs - sorted_probs) < float(top_p)
        if keep_mask.numel() > 0:
            keep_mask[0] = True
        filtered_probs = torch.where(keep_mask, sorted_probs, torch.zeros_like(sorted_probs))
        total = filtered_probs.sum()
        if float(total.item()) > 0:
            filtered_probs = filtered_probs / total
            sampled_index = int(torch.multinomial(filtered_probs, num_samples=1).item())
            return int(sorted_indices[sampled_index].item())

    return int(torch.multinomial(probs, num_samples=1).item())


def _match_stop_sequence_suffix(
    token_ids: Sequence[int],
    stop_id_sequences: Sequence[Sequence[int]],
) -> Optional[List[int]]:
    for stop_ids in stop_id_sequences:
        stop_list = list(stop_ids)
        if not stop_list or len(token_ids) < len(stop_list):
            continue
        if list(token_ids[-len(stop_list) :]) == stop_list:
            return stop_list
    return None


@torch.no_grad()
def generate_with_query_attention_trace(
    backend: GenerationBackend,
    prompt_prefix: str,
    generation_config: GenerationConfig,
    *,
    stop_strings: Optional[List[str]] = None,
) -> tuple[GenerationResult, Dict[str, Any]]:
    model = backend.model
    tokenizer = backend.tokenizer
    if model is None or tokenizer is None:
        raise ValueError("Query-attention tracing requires an HF backend with a loaded model and tokenizer.")

    prompt_token_ids = backend.encode(prompt_prefix)
    if not prompt_token_ids:
        raise ValueError("Prompt prefix tokenizes to an empty sequence.")

    stop_id_sequences = [backend.encode(item) for item in (stop_strings or []) if item]
    stop_id_sequences = [item for item in stop_id_sequences if item]

    max_new_tokens = generation_config.max_stage1_tokens if stop_id_sequences else generation_config.max_new_tokens
    model_device = get_input_device_for_model(model)

    input_ids = torch.tensor([prompt_token_ids], dtype=torch.long, device=model_device)
    attention_mask = torch.ones_like(input_ids)
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_attentions=True,
        use_cache=True,
        return_dict=True,
    )
    past_key_values = getattr(outputs, "past_key_values", None)
    if past_key_values is None:
        raise ValueError("Model did not return past_key_values during query-attention tracing.")

    generated_token_ids: List[int] = []
    step_trace: List[Dict[str, Any]] = []
    stop_reason = "max_new_tokens"
    matched_stop_ids: Optional[List[int]] = None
    captured_attentions = None
    captured_query_index: Optional[int] = None
    captured_token_id: Optional[int] = None

    next_token_logits = outputs.logits[0, -1]
    eos_token_id = tokenizer.eos_token_id
    for step_idx in range(max(int(max_new_tokens), 0)):
        next_token_id = _sample_next_token_id(
            next_token_logits,
            do_sample=generation_config.do_sample,
            temperature=generation_config.temperature,
            top_p=generation_config.top_p,
        )
        generated_token_ids.append(next_token_id)

        full_sequence_length = len(prompt_token_ids) + len(generated_token_ids)
        step_input_ids = torch.tensor([[next_token_id]], dtype=torch.long, device=model_device)
        step_attention_mask = torch.ones((1, full_sequence_length), dtype=torch.long, device=model_device)
        step_outputs = model(
            input_ids=step_input_ids,
            attention_mask=step_attention_mask,
            past_key_values=past_key_values,
            output_attentions=True,
            use_cache=True,
            return_dict=True,
        )
        past_key_values = getattr(step_outputs, "past_key_values", None)
        if past_key_values is None:
            raise ValueError("Model did not return past_key_values for an incremental generation step.")

        captured_attentions = getattr(step_outputs, "attentions", None)
        if captured_attentions is None:
            raise ValueError("Model did not return attentions during query-attention tracing.")
        captured_query_index = full_sequence_length - 1
        captured_token_id = next_token_id

        step_trace.append(
            {
                "step": int(step_idx),
                "generated_token_id": int(next_token_id),
                "generated_token_text": backend.decode([next_token_id]),
                "query_index_in_full_sequence": int(captured_query_index),
                "full_sequence_length": int(full_sequence_length),
            }
        )

        matched_stop_ids = _match_stop_sequence_suffix(generated_token_ids, stop_id_sequences)
        if matched_stop_ids is not None:
            stop_reason = "matched_stop_sequence"
            break
        if eos_token_id is not None and int(next_token_id) == int(eos_token_id):
            stop_reason = "eos_token"
            break

        next_token_logits = step_outputs.logits[0, -1]

    continuation = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
    generation_result = GenerationResult(
        prompt_prefix=prompt_prefix,
        continuation=continuation,
        full_text=prompt_prefix + continuation,
        generated_tokens=len(generated_token_ids),
        token_ids=[int(token_id) for token_id in generated_token_ids],
        step_scores=[],
    )
    trace = {
        "prompt_token_ids": [int(token_id) for token_id in prompt_token_ids],
        "stop_reason": stop_reason,
        "matched_stop_token_ids": [int(token_id) for token_id in matched_stop_ids] if matched_stop_ids else None,
        "matched_stop_text": backend.decode(matched_stop_ids) if matched_stop_ids else "",
        "captured_attentions": captured_attentions,
        "captured_query_index": captured_query_index,
        "captured_token_id": captured_token_id,
        "captured_token_text": backend.decode([captured_token_id]) if captured_token_id is not None else "",
        "step_trace": step_trace,
    }
    return generation_result, trace


@torch.no_grad()
def generate_with_next_token_attention_trace(
    backend: GenerationBackend,
    prompt_prefix: str,
    generation_config: GenerationConfig,
    *,
    stop_strings: Optional[List[str]] = None,
) -> tuple[GenerationResult, Dict[str, Any]]:
    model = backend.model
    tokenizer = backend.tokenizer
    if model is None or tokenizer is None:
        raise ValueError("Next-token attention tracing requires an HF backend with a loaded model and tokenizer.")

    prompt_token_ids = backend.encode(prompt_prefix)
    if not prompt_token_ids:
        raise ValueError("Prompt prefix tokenizes to an empty sequence.")

    stop_id_sequences = [backend.encode(item) for item in (stop_strings or []) if item]
    stop_id_sequences = [item for item in stop_id_sequences if item]

    max_new_tokens = generation_config.max_stage1_tokens if stop_id_sequences else generation_config.max_new_tokens
    model_device = get_input_device_for_model(model)

    input_ids = torch.tensor([prompt_token_ids], dtype=torch.long, device=model_device)
    attention_mask = torch.ones_like(input_ids)
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_attentions=True,
        use_cache=True,
        return_dict=True,
    )
    past_key_values = getattr(outputs, "past_key_values", None)
    if past_key_values is None:
        raise ValueError("Model did not return past_key_values during next-token attention tracing.")

    generated_token_ids: List[int] = []
    step_trace: List[Dict[str, Any]] = []
    stop_reason = "max_new_tokens"
    matched_stop_ids: Optional[List[int]] = None

    current_attentions = getattr(outputs, "attentions", None)
    if current_attentions is None:
        raise ValueError("Model did not return attentions during next-token attention tracing.")
    current_next_token_logits = outputs.logits[0, -1]
    current_prefix_token_ids = list(prompt_token_ids)

    captured_attentions = None
    captured_query_index: Optional[int] = None
    captured_predicted_token_id: Optional[int] = None
    captured_prefix_last_token_id: Optional[int] = None
    captured_prefix_last_token_text = ""

    eos_token_id = tokenizer.eos_token_id
    for step_idx in range(max(int(max_new_tokens), 0)):
        next_token_id = _sample_next_token_id(
            current_next_token_logits,
            do_sample=generation_config.do_sample,
            temperature=generation_config.temperature,
            top_p=generation_config.top_p,
        )

        captured_attentions = current_attentions
        captured_query_index = len(current_prefix_token_ids) - 1
        captured_predicted_token_id = int(next_token_id)
        captured_prefix_last_token_id = int(current_prefix_token_ids[-1])
        captured_prefix_last_token_text = backend.decode([captured_prefix_last_token_id])

        generated_token_ids.append(int(next_token_id))
        step_trace.append(
            {
                "step": int(step_idx),
                "predicted_token_id": int(next_token_id),
                "predicted_token_text": backend.decode([next_token_id]),
                "prefix_length": int(len(current_prefix_token_ids)),
                "query_index_in_full_sequence": int(captured_query_index),
                "prefix_last_token_id": int(captured_prefix_last_token_id),
                "prefix_last_token_text": captured_prefix_last_token_text,
            }
        )

        matched_stop_ids = _match_stop_sequence_suffix(generated_token_ids, stop_id_sequences)
        if matched_stop_ids is not None:
            stop_reason = "matched_stop_sequence"
            break
        if eos_token_id is not None and int(next_token_id) == int(eos_token_id):
            stop_reason = "eos_token"
            break

        current_prefix_token_ids.append(int(next_token_id))
        step_input_ids = torch.tensor([[next_token_id]], dtype=torch.long, device=model_device)
        step_attention_mask = torch.ones((1, len(current_prefix_token_ids)), dtype=torch.long, device=model_device)
        step_outputs = model(
            input_ids=step_input_ids,
            attention_mask=step_attention_mask,
            past_key_values=past_key_values,
            output_attentions=True,
            use_cache=True,
            return_dict=True,
        )
        past_key_values = getattr(step_outputs, "past_key_values", None)
        if past_key_values is None:
            raise ValueError("Model did not return past_key_values for an incremental generation step.")

        current_attentions = getattr(step_outputs, "attentions", None)
        if current_attentions is None:
            raise ValueError("Model did not return attentions during incremental next-token tracing.")
        current_next_token_logits = step_outputs.logits[0, -1]

    continuation = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
    generation_result = GenerationResult(
        prompt_prefix=prompt_prefix,
        continuation=continuation,
        full_text=prompt_prefix + continuation,
        generated_tokens=len(generated_token_ids),
        token_ids=[int(token_id) for token_id in generated_token_ids],
        step_scores=[],
    )
    trace = {
        "prompt_token_ids": [int(token_id) for token_id in prompt_token_ids],
        "stop_reason": stop_reason,
        "matched_stop_token_ids": [int(token_id) for token_id in matched_stop_ids] if matched_stop_ids else None,
        "matched_stop_text": backend.decode(matched_stop_ids) if matched_stop_ids else "",
        "captured_attentions": captured_attentions,
        "captured_query_index": captured_query_index,
        "captured_predicted_token_id": captured_predicted_token_id,
        "captured_predicted_token_text": backend.decode([captured_predicted_token_id]) if captured_predicted_token_id is not None else "",
        "captured_prefix_last_token_id": captured_prefix_last_token_id,
        "captured_prefix_last_token_text": captured_prefix_last_token_text,
        "step_trace": step_trace,
    }
    return generation_result, trace


def generate_stage1_until_first_think_close(
    backend: GenerationBackend,
    question: str,
    generation_config: GenerationConfig,
    tracked_token_ids: Optional[List[int]] = None,
) -> GenerationResult:
    prompt = backend.build_prompt(question, generation_config)
    return backend.generate(
        prompt,
        generation_config,
        stop_strings=[generation_config.stage1_stop_string] if generation_config.stage1_stop_string else None,
        tracked_token_ids=tracked_token_ids,
    )


def generate_from_prefix(
    backend: GenerationBackend,
    prompt_prefix: str,
    generation_config: GenerationConfig,
    tracked_token_ids: Optional[List[int]] = None,
) -> GenerationResult:
    return backend.generate(
        prompt_prefix,
        generation_config,
        tracked_token_ids=tracked_token_ids,
    )


def generate_full_cot(
    backend: GenerationBackend,
    question: str,
    generation_config: GenerationConfig,
    tracked_token_ids: Optional[List[int]] = None,
) -> GenerationResult:
    prompt = backend.build_prompt(question, generation_config)
    return backend.generate(
        prompt,
        generation_config,
        tracked_token_ids=tracked_token_ids,
    )
