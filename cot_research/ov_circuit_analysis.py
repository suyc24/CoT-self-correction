from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from .generation import _match_stop_sequence_suffix, _sample_next_token_id, GenerationBackend
from .model_utils import (
    get_attention_module,
    get_decoder_layers,
    get_input_device_for_model,
    get_nested_attr,
    infer_attention_head_shape,
)
from .schemas import GenerationConfig, GenerationResult


def _dedup_preserve_order(values: Sequence[int]) -> List[int]:
    seen = set()
    deduped: List[int] = []
    for value in values:
        key = int(value)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(key)
    return deduped


def _decode_token(tokenizer, token_id: int) -> str:
    try:
        return tokenizer.decode([int(token_id)], skip_special_tokens=False)
    except Exception:
        return ""


def _get_embeddings_weight(model: torch.nn.Module) -> torch.Tensor:
    candidates = [
        "model.embed_tokens.weight",
        "transformer.wte.weight",
        "model.decoder.embed_tokens.weight",
    ]
    for path in candidates:
        weight = get_nested_attr(model, path)
        if isinstance(weight, torch.Tensor):
            return weight
    raise ValueError("Cannot find model token embedding weight.")


def _get_lm_head_weight(model: torch.nn.Module) -> torch.Tensor:
    candidates = [
        "lm_head.weight",
        "model.lm_head.weight",
        "embed_out.weight",
    ]
    for path in candidates:
        weight = get_nested_attr(model, path)
        if isinstance(weight, torch.Tensor):
            return weight
    raise ValueError("Cannot find model lm_head / unembedding weight.")


def _resolve_query_head_to_kv_head(attn_module: torch.nn.Module, query_head_idx: int, num_query_heads: int) -> Dict[str, int]:
    num_kv_heads = getattr(attn_module, "num_key_value_heads", None)
    if num_kv_heads is None:
        return {
            "query_head_idx": int(query_head_idx),
            "kv_head_idx": int(query_head_idx),
            "num_query_heads": int(num_query_heads),
            "num_kv_heads": int(num_query_heads),
            "num_query_heads_per_kv_head": 1,
        }

    num_kv_heads = int(num_kv_heads)
    if num_kv_heads <= 0:
        raise ValueError(f"Invalid num_key_value_heads={num_kv_heads}.")
    if num_query_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_query_heads={num_query_heads} is not divisible by num_key_value_heads={num_kv_heads}."
        )
    query_heads_per_kv_head = num_query_heads // num_kv_heads
    kv_head_idx = int(query_head_idx // query_heads_per_kv_head)
    return {
        "query_head_idx": int(query_head_idx),
        "kv_head_idx": int(kv_head_idx),
        "num_query_heads": int(num_query_heads),
        "num_kv_heads": int(num_kv_heads),
        "num_query_heads_per_kv_head": int(query_heads_per_kv_head),
    }


def extract_head_ov_components(model: torch.nn.Module, *, layer_idx: int, head_idx: int) -> Dict[str, Any]:
    layers, layer_path = get_decoder_layers(model)
    if layer_idx < 0 or layer_idx >= len(layers):
        raise ValueError(f"Layer index {layer_idx} is out of range for {len(layers)} decoder layers.")

    layer = layers[layer_idx]
    attn_module = get_attention_module(layer)
    num_heads, head_dim = infer_attention_head_shape(model, attn_module)
    if head_idx < 0 or head_idx >= num_heads:
        raise ValueError(f"Head index {head_idx} is out of range for layer {layer_idx} with {num_heads} heads.")

    if not hasattr(layer, "input_layernorm"):
        raise ValueError("Expected decoder layer to expose input_layernorm for layer-0 OV analysis.")
    if not hasattr(attn_module, "v_proj") or not hasattr(attn_module, "o_proj"):
        raise ValueError("Attention module is missing v_proj or o_proj.")

    kv_info = _resolve_query_head_to_kv_head(attn_module, head_idx, num_heads)
    kv_head_idx = int(kv_info["kv_head_idx"])

    v_proj_weight = attn_module.v_proj.weight.detach()
    o_proj_weight = attn_module.o_proj.weight.detach()
    embed_weight = _get_embeddings_weight(model).detach()
    lm_head_weight = _get_lm_head_weight(model).detach()

    v_start = kv_head_idx * head_dim
    v_end = (kv_head_idx + 1) * head_dim
    o_start = head_idx * head_dim
    o_end = (head_idx + 1) * head_dim
    if v_proj_weight.shape[0] < v_end:
        raise ValueError(
            f"v_proj output dim {v_proj_weight.shape[0]} is smaller than requested KV slice [{v_start}:{v_end}]."
        )
    if o_proj_weight.shape[1] < o_end:
        raise ValueError(
            f"o_proj input dim {o_proj_weight.shape[1]} is smaller than requested head slice [{o_start}:{o_end}]."
        )

    return {
        "layer_path": layer_path,
        "layer_idx": int(layer_idx),
        "head_idx": int(head_idx),
        "head_label": f"L{layer_idx}H{head_idx}",
        "num_heads": int(num_heads),
        "head_dim": int(head_dim),
        "kv_head_idx": int(kv_head_idx),
        "num_kv_heads": int(kv_info["num_kv_heads"]),
        "num_query_heads_per_kv_head": int(kv_info["num_query_heads_per_kv_head"]),
        "input_layernorm": layer.input_layernorm,
        "embed_weight": embed_weight,
        "lm_head_weight": lm_head_weight,
        "v_proj_slice": v_proj_weight[v_start:v_end, :],
        "o_proj_slice": o_proj_weight[:, o_start:o_end],
    }


@torch.no_grad()
def generate_with_filtered_head_attention(
    backend: GenerationBackend,
    prompt_prefix: str,
    generation_config: GenerationConfig,
    *,
    layer_idx: int,
    head_idx: int,
    attention_threshold: float,
    stop_strings: Optional[List[str]] = None,
) -> Tuple[GenerationResult, Dict[str, Any]]:
    model = backend.model
    tokenizer = backend.tokenizer
    if model is None or tokenizer is None:
        raise ValueError("Dynamic OV analysis requires an HF backend with a loaded model and tokenizer.")

    prompt_token_ids = [int(token_id) for token_id in backend.encode(prompt_prefix)]
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
        raise ValueError("Model did not return past_key_values during dynamic OV analysis generation.")

    generated_token_ids: List[int] = []
    filtered_pair_rows: List[Dict[str, Any]] = []
    step_summaries: List[Dict[str, Any]] = []
    stop_reason = "max_new_tokens"
    matched_stop_ids: Optional[List[int]] = None

    next_token_logits = outputs.logits[0, -1]
    eos_token_id = tokenizer.eos_token_id
    full_token_ids = list(prompt_token_ids)

    for step_idx in range(max(int(max_new_tokens), 0)):
        next_token_id = _sample_next_token_id(
            next_token_logits,
            do_sample=generation_config.do_sample,
            temperature=generation_config.temperature,
            top_p=generation_config.top_p,
        )
        generated_token_ids.append(int(next_token_id))
        full_token_ids.append(int(next_token_id))
        full_sequence_length = len(full_token_ids)

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

        attentions = getattr(step_outputs, "attentions", None)
        if attentions is None:
            raise ValueError("Model did not return attentions for an incremental generation step.")
        if layer_idx < 0 or layer_idx >= len(attentions):
            raise ValueError(f"Requested layer {layer_idx} exceeds returned attention layers={len(attentions)}.")

        layer_attention = attentions[layer_idx][0].detach().float().cpu()
        if head_idx < 0 or head_idx >= int(layer_attention.shape[0]):
            raise ValueError(
                f"Requested head {head_idx} exceeds returned heads={int(layer_attention.shape[0])} for layer {layer_idx}."
            )
        head_attention = layer_attention[head_idx, -1]
        query_index = full_sequence_length - 1
        top_source_index = int(torch.argmax(head_attention).item()) if head_attention.numel() > 0 else -1
        top_attention = float(head_attention[top_source_index].item()) if top_source_index >= 0 else 0.0
        selected_source_indices = torch.nonzero(head_attention >= float(attention_threshold), as_tuple=False).view(-1).tolist()

        step_summaries.append(
            {
                "step_idx": int(step_idx),
                "query_index": int(query_index),
                "query_token_id": int(next_token_id),
                "query_token_text": _decode_token(tokenizer, int(next_token_id)),
                "top_source_index": int(top_source_index),
                "top_attention": float(top_attention),
                "selected_pair_count": int(len(selected_source_indices)),
            }
        )
        for source_index in selected_source_indices:
            source_index_int = int(source_index)
            source_token_id = int(full_token_ids[source_index_int])
            relative_distance = int(query_index - source_index_int)
            filtered_pair_rows.append(
                {
                    "step_idx": int(step_idx),
                    "query_index": int(query_index),
                    "query_token_id": int(next_token_id),
                    "query_token_text": _decode_token(tokenizer, int(next_token_id)),
                    "source_index": int(source_index_int),
                    "source_token_id": int(source_token_id),
                    "source_token_text": _decode_token(tokenizer, int(source_token_id)),
                    "attention": float(head_attention[source_index_int].item()),
                    "relative_distance": int(relative_distance),
                    "is_self_source": bool(source_index_int == query_index),
                    "is_prev_1_source": bool(relative_distance == 1),
                    "is_prompt_source": bool(source_index_int < len(prompt_token_ids)),
                    "is_generated_source": bool(source_index_int >= len(prompt_token_ids)),
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
        "filtered_pair_rows": filtered_pair_rows,
        "step_summaries": step_summaries,
        "stop_reason": stop_reason,
        "matched_stop_token_ids": [int(token_id) for token_id in matched_stop_ids] if matched_stop_ids else None,
        "matched_stop_text": backend.decode(matched_stop_ids) if matched_stop_ids else "",
        "attention_threshold": float(attention_threshold),
    }
    return generation_result, trace


@torch.no_grad()
def compute_static_ov_metrics_for_source_tokens(
    model: torch.nn.Module,
    tokenizer,
    *,
    layer_idx: int,
    head_idx: int,
    source_token_ids: Sequence[int],
    score_batch_size: int = 256,
    top_k_suppressed: int = 10,
    bottom_fraction: float = 0.05,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    source_token_ids = _dedup_preserve_order([int(token_id) for token_id in source_token_ids])
    if not source_token_ids:
        return [], {
            "observed_source_token_type_count": 0,
            "negative_self_logit_rate": 0.0,
            "topk_suppressed_rate": 0.0,
            "bottom_fraction_rate": 0.0,
        }

    components = extract_head_ov_components(model, layer_idx=layer_idx, head_idx=head_idx)
    model_device = get_input_device_for_model(model)
    input_layernorm = components["input_layernorm"]
    embed_weight = components["embed_weight"].to(model_device)
    lm_head_weight = components["lm_head_weight"].to(device=model_device, dtype=torch.float32)
    v_proj_slice = components["v_proj_slice"].to(device=model_device, dtype=torch.float32)
    o_proj_slice = components["o_proj_slice"].to(device=model_device, dtype=torch.float32)

    projected_unembedding = torch.matmul(lm_head_weight, o_proj_slice)
    vocab_size = int(projected_unembedding.shape[0])
    top_k_suppressed = max(1, min(int(top_k_suppressed), vocab_size))
    bottom_rank_cutoff = max(1, int(math.ceil(float(vocab_size) * float(bottom_fraction))))

    rows: List[Dict[str, Any]] = []
    for start in range(0, len(source_token_ids), max(int(score_batch_size), 1)):
        batch_token_ids = source_token_ids[start : start + max(int(score_batch_size), 1)]
        token_id_tensor = torch.tensor(batch_token_ids, dtype=torch.long, device=model_device)
        source_embeddings = embed_weight.index_select(0, token_id_tensor)
        source_hidden = input_layernorm(source_embeddings).to(dtype=torch.float32)
        source_latents = torch.matmul(source_hidden, v_proj_slice.transpose(0, 1))
        score_matrix = torch.matmul(projected_unembedding, source_latents.transpose(0, 1))
        self_scores = score_matrix[token_id_tensor, torch.arange(len(batch_token_ids), device=model_device)]
        self_ranks_from_bottom = (score_matrix < self_scores.unsqueeze(0)).sum(dim=0) + 1
        top_values, top_indices = torch.topk(score_matrix, k=top_k_suppressed, largest=False, dim=0)
        score_means = score_matrix.mean(dim=0)
        score_stds = score_matrix.std(dim=0, unbiased=False)

        for col_idx, token_id in enumerate(batch_token_ids):
            suppressed_ids = [int(item) for item in top_indices[:, col_idx].tolist()]
            suppressed_scores = [float(item) for item in top_values[:, col_idx].tolist()]
            rank_from_bottom = int(self_ranks_from_bottom[col_idx].item())
            self_score = float(self_scores[col_idx].item())
            rows.append(
                {
                    "source_token_id": int(token_id),
                    "source_token_text": _decode_token(tokenizer, int(token_id)),
                    "self_logit_contribution": self_score,
                    "self_rank_from_bottom": rank_from_bottom,
                    "self_percentile_from_bottom": float(rank_from_bottom / max(vocab_size, 1)),
                    "self_is_negative": bool(self_score < 0.0),
                    "self_is_topk_suppressed": bool(rank_from_bottom <= top_k_suppressed),
                    "self_is_bottom_fraction_suppressed": bool(rank_from_bottom <= bottom_rank_cutoff),
                    "score_mean_over_vocab": float(score_means[col_idx].item()),
                    "score_std_over_vocab": float(score_stds[col_idx].item()),
                    "top_suppressed_token_ids": suppressed_ids,
                    "top_suppressed_token_texts": [_decode_token(tokenizer, item) for item in suppressed_ids],
                    "top_suppressed_scores": suppressed_scores,
                }
            )
        del token_id_tensor
        del source_embeddings
        del source_hidden
        del source_latents
        del score_matrix
        del self_scores
        del self_ranks_from_bottom
        del top_values
        del top_indices
        del score_means
        del score_stds
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    negative_rate = sum(1 for row in rows if bool(row["self_is_negative"])) / len(rows)
    topk_rate = sum(1 for row in rows if bool(row["self_is_topk_suppressed"])) / len(rows)
    bottom_fraction_rate = sum(1 for row in rows if bool(row["self_is_bottom_fraction_suppressed"])) / len(rows)
    summary = {
        "observed_source_token_type_count": int(len(rows)),
        "negative_self_logit_rate": float(round(negative_rate, 6)),
        "topk_suppressed_rate": float(round(topk_rate, 6)),
        "bottom_fraction_rate": float(round(bottom_fraction_rate, 6)),
        "top_k_suppressed": int(top_k_suppressed),
        "bottom_fraction": float(bottom_fraction),
        "vocab_size": int(vocab_size),
        "kv_head_idx": int(components["kv_head_idx"]),
        "num_kv_heads": int(components["num_kv_heads"]),
        "num_query_heads_per_kv_head": int(components["num_query_heads_per_kv_head"]),
    }
    return rows, summary
