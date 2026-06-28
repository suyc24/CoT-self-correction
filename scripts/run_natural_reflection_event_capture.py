#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from cot_research.answer_extraction import answers_match, extract_last_boxed
from cot_research.cot_editing import find_last_boxed_span
from cot_research.generation import create_backend
from cot_research.hidden_trajectory import forward_one_with_boundary_hooks, prefill_before_final_full_ids
from cot_research.io_utils import dump_jsonl, load_jsonl, write_json
from cot_research.model_utils import get_decoder_layers
from cot_research.runtime_utils import seed_everything
from cot_research.schemas import BackendConfig, GenerationConfig

MARKER_PATTERNS = [
    ("wait", re.compile(r"(?i)(?<![a-z])wait(?![a-z])")),
    ("actually", re.compile(r"(?i)(?<![a-z])actually(?![a-z])")),
    ("however", re.compile(r"(?i)(?<![a-z])however(?![a-z])")),
    ("but", re.compile(r"(?i)(?<![a-z])but(?![a-z])")),
    ("check", re.compile(r"(?i)(?<![a-z])(?:check|recheck|verify)(?![a-z])")),
    ("wrong", re.compile(r"(?i)(?<![a-z])(?:wrong|incorrect|mistake|inconsistent)(?![a-z])")),
    ("hold_on", re.compile(r"(?i)hold on|let me check|reconsider")),
    ("cn_wrong", re.compile(r"不对|错误|矛盾|有误")),
    ("cn_recheck", re.compile(r"重新|检查|重算|再算|等等|等一下")),
]
EXPLICIT_ERROR_PATTERNS = [re.compile(r"(?i)wrong|incorrect|mistake|not correct|inconsistent|error"), re.compile(r"不对|错误|矛盾|有误")]
REPAIR_PATTERNS = [re.compile(r"(?i)recompute|recalculate|calculate|check|verify|instead|so the correct"), re.compile(r"重新|重算|检查|改为|应该是")]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Capture reflection hidden states during first-pass and post-tamper generation.")
    p.add_argument("--input_jsonl", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--model_name_or_path", default="Qwen/Qwen3-4B")
    p.add_argument("--start_idx", type=int, default=0)
    p.add_argument("--max_examples", type=int, default=20)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--device_map", default="")
    p.add_argument("--load_in_half", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--use_fast_tokenizer", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--use_safetensors", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--attn_implementation", default="eager")
    p.add_argument("--system_prompt", default="Please reason step by step, and put your final answer within \\boxed{}.")
    p.add_argument("--assistant_prefix", default="")
    p.add_argument("--max_stage1_tokens", type=int, default=16384)
    p.add_argument("--max_intervention_tokens", type=int, default=512)
    p.add_argument("--stop_at_think_end", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--stage1_stop_string", default="</think>")
    p.add_argument("--layers", default="19-22")
    p.add_argument("--sites", default="post_attn,block_output")
    p.add_argument("--do_sample", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--event_future_chars", type=int, default=800)
    p.add_argument("--save_full_text", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--print_every", type=int, default=1)
    return p.parse_args()


def parse_layer_spec(text: str, n_layers: int) -> List[int]:
    out: List[int] = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        values = []
        if "-" in item:
            a, b = item.split("-", 1)
            start, end = int(a), int(b)
            step = 1 if end >= start else -1
            values = list(range(start, end + step, step))
        else:
            values = [int(item)]
        for idx in values:
            if idx < 0:
                idx = n_layers + idx
            if idx < 0 or idx >= n_layers:
                raise ValueError(f"Layer index out of range: {idx}")
            if idx not in out:
                out.append(idx)
    if not out:
        raise ValueError("No layers selected.")
    return out


def parse_csv_list(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def build_backend(args: argparse.Namespace):
    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu_id))
    device_map: Any = {"": int(args.gpu_id)}
    if str(args.device_map).strip():
        device_map = args.device_map
    return create_backend(BackendConfig(
        backend_type="hf",
        model_name_or_path=args.model_name_or_path,
        device_map=device_map,
        load_in_half=bool(args.load_in_half),
        use_fast_tokenizer=bool(args.use_fast_tokenizer),
        use_safetensors=bool(args.use_safetensors),
        local_files_only=bool(args.local_files_only),
        attn_implementation=args.attn_implementation,
    ))


def generation_config(args: argparse.Namespace) -> GenerationConfig:
    return GenerationConfig(system_prompt=args.system_prompt, assistant_prefix=args.assistant_prefix, max_stage1_tokens=int(args.max_stage1_tokens), max_new_tokens=int(args.max_intervention_tokens), do_sample=bool(args.do_sample), temperature=float(args.temperature), top_p=float(args.top_p))


def row_id(row: Dict[str, Any], fallback: int) -> str:
    return str(row.get("id") or row.get("unique_id") or row.get("problem_id") or fallback)


def row_question(row: Dict[str, Any]) -> str:
    return str(row.get("question") or row.get("problem") or "").strip()


def row_answer(row: Dict[str, Any]) -> str:
    return str(row.get("correct_answer") or row.get("answer") or "").strip()


def sha1_text(text: str) -> str:
    return hashlib.sha1(str(text).encode("utf-8", errors="replace")).hexdigest()


def choose_next_token_id(logits: torch.Tensor, *, do_sample: bool, temperature: float, top_p: float) -> int:
    logits = logits[0] if logits.ndim == 2 else logits
    if not do_sample:
        return int(torch.argmax(logits).item())
    scaled = logits.float() / max(float(temperature), 1e-6)
    probs = torch.softmax(scaled, dim=-1)
    if 0 < float(top_p) < 1:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        keep = cumulative <= float(top_p)
        if keep.numel():
            keep[0] = True
        filtered = sorted_probs * keep
        filtered = filtered / filtered.sum().clamp_min(1e-12)
        choice = torch.multinomial(filtered, num_samples=1)
        return int(sorted_idx[int(choice.item())].item())
    return int(torch.multinomial(probs, num_samples=1).item())


def marker_kind(window: str) -> Optional[str]:
    for name, pattern in MARKER_PATTERNS:
        if pattern.search(window):
            return name
    return None


def newly_completed_marker(prev_text: str, text: str) -> Optional[str]:
    boundary = len(prev_text)
    for name, pattern in MARKER_PATTERNS:
        start = max(0, len(text) - 120)
        for match in pattern.finditer(text[start:]):
            abs_start = start + match.start()
            abs_end = start + match.end()
            if abs_start < len(text) and abs_end >= boundary:
                return name
    return None


def find_think_span(text: str) -> Tuple[int, int]:
    start = text.find("<think>")
    start = start + len("<think>") if start >= 0 else 0
    end = text.find("</think>", start)
    return start, (end if end >= 0 else len(text))


def replace_last_boxed_in_cot(text: str, replacement: str) -> Tuple[Optional[str], Optional[str]]:
    start, end = find_think_span(text)
    think_text = text[start:end]
    span = find_last_boxed_span(think_text)
    if span is None:
        span = find_last_boxed_span(text)
        if span is None:
            return None, None
        abs_start, abs_end = span
    else:
        abs_start, abs_end = start + span[0], start + span[1]
    old = text[abs_start:abs_end]
    if not old.startswith("\\boxed{") or not old.endswith("}"):
        return None, None
    return text[:abs_start] + f"\\boxed{{{replacement}}}", old


def wrong_answer_for_row(row: Dict[str, Any], baseline_answer: str) -> str:
    for key in ("wrong_answer", "tamper_answer"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    answer = str(baseline_answer or row_answer(row) or "").strip()
    m = re.search(r"-?\d+", answer)
    if m:
        try:
            return answer[:m.start()] + str(int(m.group(0)) + 1) + answer[m.end():]
        except ValueError:
            pass
    return "0" if answer != "0" else "1"


def has_any(patterns: Sequence[re.Pattern[str]], text: str) -> bool:
    return any(p.search(text) is not None for p in patterns)


def summarize_future(full_text: str, event_char_end: int, *, answer: str, tampered_answer: str, future_chars: int) -> Dict[str, Any]:
    future = full_text[int(event_char_end): int(event_char_end) + int(future_chars)]
    boxed = extract_last_boxed(full_text) or ""
    return {
        "future_explicit_error_ack": has_any(EXPLICIT_ERROR_PATTERNS, future),
        "future_repair_language": has_any(REPAIR_PATTERNS, future),
        "final_boxed_answer": boxed,
        "final_answer_correct": bool(boxed and answer and answers_match(boxed, answer)),
        "final_answer_matches_tamper": bool(boxed and tampered_answer and answers_match(boxed, tampered_answer)),
    }


def init_feature_store(layer_indices: Sequence[int], sites: Sequence[str]) -> Dict[str, Dict[str, List[torch.Tensor]]]:
    return {f"L{int(layer_idx)}/{site}": {kind: [] for kind in ("h_pre", "h_marker", "h_post", "delta_marker", "delta_post")} for layer_idx in layer_indices for site in sites}


def append_features(store: Dict[str, Dict[str, List[torch.Tensor]]], *, pre: Dict[Tuple[str, int], torch.Tensor], marker: Dict[Tuple[str, int], torch.Tensor], post: Optional[Dict[Tuple[str, int], torch.Tensor]], layer_indices: Sequence[int], sites: Sequence[str]) -> bool:
    ok = False
    for layer_idx in layer_indices:
        for site in sites:
            a = pre.get((site, int(layer_idx)))
            b = marker.get((site, int(layer_idx)))
            if a is None or b is None:
                continue
            c = post.get((site, int(layer_idx))) if post is not None else b
            if c is None:
                c = b
            key = f"L{int(layer_idx)}/{site}"
            store[key]["h_pre"].append(a.detach().cpu().float())
            store[key]["h_marker"].append(b.detach().cpu().float())
            store[key]["h_post"].append(c.detach().cpu().float())
            store[key]["delta_marker"].append((b - a).detach().cpu().float())
            store[key]["delta_post"].append((c - b).detach().cpu().float())
            ok = True
    return ok


def stack_features(store: Dict[str, Dict[str, List[torch.Tensor]]]) -> Dict[str, Dict[str, torch.Tensor]]:
    out: Dict[str, Dict[str, torch.Tensor]] = {}
    for key, kinds in store.items():
        out[key] = {}
        for kind, values in kinds.items():
            out[key][kind] = torch.stack(values, dim=0).to(torch.float16) if values else torch.empty((0, 0), dtype=torch.float16)
    return out


@torch.no_grad()
def generate_with_event_capture(model: torch.nn.Module, tokenizer, *, prompt_text: str, layers: Sequence[torch.nn.Module], layer_indices: Sequence[int], sites: Sequence[str], max_new_tokens: int, do_sample: bool, temperature: float, top_p: float, stop_string: str, stop_at_stop_string: bool, scope: str, example_meta: Dict[str, Any], feature_store: Dict[str, Dict[str, List[torch.Tensor]]], event_rows: List[Dict[str, Any]], future_chars: int) -> Dict[str, Any]:
    input_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    if len(input_ids) < 2:
        raise ValueError("Prompt produced fewer than two token ids.")
    past, ids_before_final, final_token_id = prefill_before_final_full_ids(model, input_ids)
    past, logits, prev_captures, _ = forward_one_with_boundary_hooks(model, past=past, full_ids_before_token=ids_before_final, token_id=final_token_id, layers=layers, capture_layer_indices=layer_indices, capture_sites=sites, capture_dtype=torch.float16)
    full_ids = [int(x) for x in input_ids]
    generated: List[int] = []
    captures_by_step: List[Dict[Tuple[str, int], torch.Tensor]] = []
    pending: List[Dict[str, Any]] = []
    stop_reason = "max_new_tokens"
    eos_id = tokenizer.eos_token_id
    first_event_row = len(event_rows)

    for step_idx in range(int(max_new_tokens)):
        next_id = choose_next_token_id(logits, do_sample=do_sample, temperature=temperature, top_p=top_p)
        prefix_before_next = list(full_ids)
        generated.append(int(next_id))
        full_ids.append(int(next_id))
        past, logits, marker_captures, _ = forward_one_with_boundary_hooks(model, past=past, full_ids_before_token=prefix_before_next, token_id=int(next_id), layers=layers, capture_layer_indices=layer_indices, capture_sites=sites, capture_dtype=torch.float16)
        captures_by_step.append(marker_captures)

        prev_text = tokenizer.decode(generated[:-1], skip_special_tokens=False)
        text = tokenizer.decode(generated, skip_special_tokens=False)
        token_text = tokenizer.decode([int(next_id)], skip_special_tokens=False)
        token_char_end = len(text)
        token_char_start = max(0, token_char_end - len(token_text))
        kind = newly_completed_marker(prev_text, text)
        if kind:
            pending.append({"event_step": int(step_idx), "event_char_start": int(token_char_start), "event_char_end": int(token_char_end), "marker_kind": kind, "marker_token_id": int(next_id), "marker_token_text": token_text, "pre": prev_captures, "marker": marker_captures})

        ready, keep = [], []
        for item in pending:
            (ready if int(step_idx) >= int(item["event_step"]) + 1 else keep).append(item)
        pending = keep
        for item in ready:
            post = captures_by_step[int(item["event_step"]) + 1]
            if append_features(feature_store, pre=item["pre"], marker=item["marker"], post=post, layer_indices=layer_indices, sites=sites):
                event_rows.append({"event_index": len(event_rows), **example_meta, "scope": scope, "marker_kind": item["marker_kind"], "marker_token_id": item["marker_token_id"], "marker_token_text": item["marker_token_text"], "event_step": item["event_step"], "event_char_start": item["event_char_start"], "event_char_end": item["event_char_end"]})

        if stop_at_stop_string and stop_string and stop_string in text:
            stop_reason = "matched_stop_string"
            break
        if eos_id is not None and int(next_id) == int(eos_id):
            stop_reason = "eos_token"
            break
        prev_captures = marker_captures

    final_post = captures_by_step[-1] if captures_by_step else prev_captures
    for item in pending:
        if append_features(feature_store, pre=item["pre"], marker=item["marker"], post=final_post, layer_indices=layer_indices, sites=sites):
            event_rows.append({"event_index": len(event_rows), **example_meta, "scope": scope, "marker_kind": item["marker_kind"], "marker_token_id": item["marker_token_id"], "marker_token_text": item["marker_token_text"], "event_step": item["event_step"], "event_char_start": item["event_char_start"], "event_char_end": item["event_char_end"]})

    generated_text = tokenizer.decode(generated, skip_special_tokens=False)
    full_text = prompt_text + generated_text
    for row in event_rows[first_event_row:]:
        row.update(summarize_future(full_text, len(prompt_text) + int(row.get("event_char_end", 0)), answer=str(example_meta.get("correct_answer") or ""), tampered_answer=str(example_meta.get("tampered_answer") or ""), future_chars=future_chars))
        row["relative_position"] = float(row.get("event_step", 0)) / max(len(generated), 1)
    return {"generated_token_ids": generated, "generated_text": generated_text, "full_text": full_text, "stop_reason": stop_reason, "generated_tokens": len(generated)}


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key); fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def main() -> None:
    args = parse_args()
    seed_everything(int(args.seed))
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(args.input_jsonl)
    selected = rows[int(args.start_idx): int(args.start_idx) + int(args.max_examples)]
    backend = build_backend(args)
    if backend.model is None:
        raise RuntimeError("HF model required.")
    model, tokenizer = backend.model, backend.tokenizer
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    else:
        layers = get_decoder_layers(model)
    layer_indices = parse_layer_spec(args.layers, len(layers))
    sites = parse_csv_list(args.sites)
    gen_cfg = generation_config(args)
    features = init_feature_store(layer_indices, sites)
    events: List[Dict[str, Any]] = []
    behaviors: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    completed_path = out / "completed_ids.jsonl"
    completed = set()
    if args.resume and completed_path.exists():
        for line in completed_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                completed.add(json.loads(line).get("example_id"))

    for local_idx, row in enumerate(tqdm(selected, desc="Natural reflection capture")):
        meta_global = row.get("level5_index") or (row.get("metadata") or {}).get("level5_index")
        global_idx = int(meta_global) if meta_global is not None else int(args.start_idx) + int(local_idx)
        ex_id = row_id(row, global_idx)
        if args.resume and ex_id in completed:
            continue
        q, correct = row_question(row), row_answer(row)
        if not q:
            skipped.append({"example_id": ex_id, "global_idx": global_idx, "reason": "empty_question"}); continue
        prompt = backend.build_prompt(q, gen_cfg)
        meta = {"example_id": ex_id, "global_idx": global_idx, "question_hash": sha1_text(q), "correct_answer": correct, "subject": row.get("metadata", {}).get("subject") or row.get("subject") or "", "level": row.get("metadata", {}).get("level") or row.get("level") or ""}
        try:
            before = len(events)
            base = generate_with_event_capture(model, tokenizer, prompt_text=prompt, layers=layers, layer_indices=layer_indices, sites=sites, max_new_tokens=int(args.max_stage1_tokens), do_sample=bool(args.do_sample), temperature=float(args.temperature), top_p=float(args.top_p), stop_string=str(args.stage1_stop_string), stop_at_stop_string=bool(args.stop_at_think_end), scope="natural_baseline", example_meta={**meta, "tampered_answer": ""}, feature_store=features, event_rows=events, future_chars=int(args.event_future_chars))
            base_box = extract_last_boxed(base["generated_text"]) or ""
            tamper_ans = wrong_answer_for_row(row, base_box or correct)
            tampered_text, old_box = replace_last_boxed_in_cot(base["generated_text"], tamper_ans)
            tamper_status, tamper_tokens, tamper_before = "not_run", 0, len(events)
            if tampered_text is None:
                skipped.append({"example_id": ex_id, "global_idx": global_idx, "reason": "no_boxed_answer_for_tamper"})
            else:
                tamper_status = "ok"
                cont = generate_with_event_capture(model, tokenizer, prompt_text=prompt + tampered_text, layers=layers, layer_indices=layer_indices, sites=sites, max_new_tokens=int(args.max_intervention_tokens), do_sample=bool(args.do_sample), temperature=float(args.temperature), top_p=float(args.top_p), stop_string="", stop_at_stop_string=False, scope="tampered_continuation", example_meta={**meta, "tampered_answer": tamper_ans}, feature_store=features, event_rows=events, future_chars=int(args.event_future_chars))
                tamper_tokens = int(cont["generated_tokens"])
            behaviors.append({"example_id": ex_id, "global_idx": global_idx, "baseline_generated_tokens": int(base["generated_tokens"]), "baseline_stop_reason": base["stop_reason"], "baseline_event_count": len(events) - before - (len(events) - tamper_before), "baseline_boxed_answer": base_box, "baseline_answer_correct": bool(base_box and correct and answers_match(base_box, correct)), "tamper_status": tamper_status, "tampered_old_box": old_box or "", "tampered_answer": tamper_ans, "tampered_generated_tokens": tamper_tokens, "tampered_event_count": len(events) - tamper_before})
            if bool(args.save_full_text):
                per = out / "per_example"; per.mkdir(parents=True, exist_ok=True)
                (per / f"{global_idx:05d}_{sha1_text(ex_id)[:10]}.json").write_text(json.dumps({"example_id": ex_id, "prompt": prompt, "baseline_generated_text": base["generated_text"], "tampered_cot_text": tampered_text or ""}, ensure_ascii=False, indent=2), encoding="utf-8")
            with completed_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"example_id": ex_id}, ensure_ascii=False) + "\n")
        except Exception as exc:
            skipped.append({"example_id": ex_id, "global_idx": global_idx, "reason": type(exc).__name__, "message": str(exc)})
        if int(args.print_every) > 0 and (len(behaviors) + len(skipped)) % int(args.print_every) == 0:
            dump_jsonl(out / "behavior_rows.partial.jsonl", behaviors)
            dump_jsonl(out / "event_rows.partial.jsonl", events)
            dump_jsonl(out / "skipped_rows.partial.jsonl", skipped)
            torch.save({"features": stack_features(features)}, out / "event_features.partial.pt")

    dump_jsonl(out / "behavior_rows.jsonl", behaviors)
    dump_jsonl(out / "event_rows.jsonl", events)
    dump_jsonl(out / "skipped_rows.jsonl", skipped)
    torch.save({"features": stack_features(features), "layers": layer_indices, "sites": sites, "feature_kinds": ["h_pre", "h_marker", "h_post", "delta_marker", "delta_post"]}, out / "event_features.pt")
    write_csv(out / "behavior_summary.csv", behaviors)
    write_json(out / "summary.json", {"input_jsonl": str(args.input_jsonl), "output_dir": str(out), "start_idx": int(args.start_idx), "max_examples": int(args.max_examples), "attempted": len(selected), "completed": len(behaviors), "skipped": len(skipped), "events": len(events), "event_scope_counts": dict(Counter(str(r.get("scope")) for r in events)), "marker_kind_counts": dict(Counter(str(r.get("marker_kind")) for r in events)), "layers": layer_indices, "sites": sites, "max_stage1_tokens": int(args.max_stage1_tokens), "max_intervention_tokens": int(args.max_intervention_tokens)})
    print(f"[Done] completed={len(behaviors)} skipped={len(skipped)} events={len(events)} output={out}")


if __name__ == "__main__":
    main()
