#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


HELPER = r'''

# --- Codex constant steering patch start ---
import atexit
import json
import os
import time

try:
    from vllm.forward_context import get_forward_context, is_forward_context_available
except Exception:  # pragma: no cover - defensive for non-vLLM imports
    get_forward_context = None

    def is_forward_context_available() -> bool:
        return False


_QWEN2_CONSTANT_STEERING_CACHE = {"path": None, "tensor": None}
_QWEN2_CONSTANT_STEERING_STATS = {
    "pid": os.getpid(),
    "total_calls": 0,
    "enabled_calls": 0,
    "matched_layer_site_calls": 0,
    "no_decode_rows_calls": 0,
    "applied_calls": 0,
    "applied_rows": 0,
    "last_total_tokens": None,
    "last_decode_rows": None,
    "last_vector_norm": None,
    "last_layer_idx": None,
    "last_site": None,
}


def _qwen2_constant_steering_debug_flush(force: bool = False) -> None:
    path = os.environ.get("VLLM_QWEN2_STEERING_DEBUG_PATH", "")
    if not path:
        return
    stats = _QWEN2_CONSTANT_STEERING_STATS
    if not force and int(stats.get("applied_calls") or 0) % 128 != 0:
        return
    try:
        payload = dict(stats)
        payload.update(
            {
                "time": time.time(),
                "enabled_env": os.environ.get("VLLM_QWEN2_STEERING_ENABLED", ""),
                "layer_env": os.environ.get("VLLM_QWEN2_STEERING_LAYER", ""),
                "site_env": os.environ.get("VLLM_QWEN2_STEERING_SITE", ""),
                "vector_path_env": os.environ.get("VLLM_QWEN2_STEERING_VECTOR_PATH", ""),
                "decode_only_env": os.environ.get("VLLM_QWEN2_STEERING_DECODE_ONLY", ""),
            }
        )
        debug_path = f"{path}.{os.getpid()}.json"
        with open(debug_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
    except Exception:
        pass


atexit.register(lambda: _qwen2_constant_steering_debug_flush(force=True))


def _qwen2_constant_steering_debug_enabled() -> bool:
    return bool(os.environ.get("VLLM_QWEN2_STEERING_DEBUG_PATH", ""))


def _qwen2_constant_steering_enabled() -> bool:
    return os.environ.get("VLLM_QWEN2_STEERING_ENABLED", "").lower() in {"1", "true", "yes", "on"}


def _qwen2_constant_steering_decode_rows(total_tokens: int) -> int:
    if os.environ.get("VLLM_QWEN2_STEERING_DECODE_ONLY", "1").lower() in {"0", "false", "no", "off"}:
        return int(total_tokens)
    if total_tokens <= 0:
        return 0
    if get_forward_context is None or not is_forward_context_available():
        return 1 if total_tokens == 1 else 0
    try:
        metadata = get_forward_context().attn_metadata
        if isinstance(metadata, list):
            metadata = metadata[0] if metadata else None
        if isinstance(metadata, dict):
            metadata = next(iter(metadata.values())) if metadata else None
        if metadata is None:
            return 1 if total_tokens == 1 else 0
        num_decode_tokens = getattr(metadata, "num_decode_tokens", None)
        if num_decode_tokens is not None:
            return min(int(num_decode_tokens), int(total_tokens))
        query_start_loc = getattr(metadata, "query_start_loc", None)
        if query_start_loc is not None and len(query_start_loc) >= 2:
            query_lens = (query_start_loc[1:] - query_start_loc[:-1]).detach().cpu().tolist()
            decode_rows = 0
            # vLLM V1 schedules decode tokens first, followed by prefill tokens.
            for value in query_lens:
                if int(value) == 1:
                    decode_rows += 1
                else:
                    break
            return min(int(decode_rows), int(total_tokens))
    except Exception:
        return 1 if total_tokens == 1 else 0
    return 1 if total_tokens == 1 else 0


def _qwen2_constant_steering_vector(hidden_states: torch.Tensor) -> torch.Tensor | None:
    vector_path = os.environ.get("VLLM_QWEN2_STEERING_VECTOR_PATH", "")
    if not vector_path:
        return None
    cache = _QWEN2_CONSTANT_STEERING_CACHE
    if cache.get("path") != vector_path:
        loaded = torch.load(vector_path, map_location="cpu")
        if isinstance(loaded, dict):
            loaded = loaded.get("vector", loaded.get("delta", loaded.get("direction")))
        if loaded is None:
            raise ValueError(f"No tensor found in steering vector file: {vector_path}")
        cache["path"] = vector_path
        cache["tensor"] = loaded.detach().float().cpu()
    vector = cache["tensor"]
    if vector is None:
        return None
    return vector.to(device=hidden_states.device, dtype=hidden_states.dtype)


def _qwen2_apply_constant_steering(
    hidden_states: torch.Tensor,
    layer_idx: int,
    site: str,
) -> torch.Tensor:
    debug_enabled = _qwen2_constant_steering_debug_enabled()
    stats = _QWEN2_CONSTANT_STEERING_STATS
    if debug_enabled:
        stats["total_calls"] = int(stats.get("total_calls") or 0) + 1
    if not _qwen2_constant_steering_enabled():
        if debug_enabled:
            _qwen2_constant_steering_debug_flush()
        return hidden_states
    if debug_enabled:
        stats["enabled_calls"] = int(stats.get("enabled_calls") or 0) + 1
    if int(os.environ.get("VLLM_QWEN2_STEERING_LAYER", "-1")) != int(layer_idx):
        if debug_enabled:
            _qwen2_constant_steering_debug_flush()
        return hidden_states
    if os.environ.get("VLLM_QWEN2_STEERING_SITE", "") != site:
        if debug_enabled:
            _qwen2_constant_steering_debug_flush()
        return hidden_states
    if debug_enabled:
        stats["matched_layer_site_calls"] = int(stats.get("matched_layer_site_calls") or 0) + 1
        stats["last_layer_idx"] = int(layer_idx)
        stats["last_site"] = str(site)
        stats["last_total_tokens"] = int(hidden_states.shape[0])
    decode_rows = _qwen2_constant_steering_decode_rows(int(hidden_states.shape[0]))
    if debug_enabled:
        stats["last_decode_rows"] = int(decode_rows)
    if decode_rows <= 0:
        if debug_enabled:
            stats["no_decode_rows_calls"] = int(stats.get("no_decode_rows_calls") or 0) + 1
            _qwen2_constant_steering_debug_flush()
        return hidden_states
    vector = _qwen2_constant_steering_vector(hidden_states)
    if vector is None:
        if debug_enabled:
            _qwen2_constant_steering_debug_flush()
        return hidden_states
    if int(vector.numel()) != int(hidden_states.shape[-1]):
        raise ValueError(
            f"Steering vector width {vector.numel()} does not match hidden size {hidden_states.shape[-1]}"
        )
    out = hidden_states.clone()
    out[:decode_rows] = out[:decode_rows] + vector.view(1, -1)
    if debug_enabled:
        stats["applied_calls"] = int(stats.get("applied_calls") or 0) + 1
        stats["applied_rows"] = int(stats.get("applied_rows") or 0) + int(decode_rows)
        try:
            stats["last_vector_norm"] = float(vector.float().norm().item())
        except Exception:
            pass
        _qwen2_constant_steering_debug_flush()
    return out

# --- Codex constant steering patch end ---
'''


def patch_qwen2_file(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    backup = path.with_suffix(path.suffix + ".codex_bak")
    if not backup.exists():
        shutil.copy2(path, backup)

    marker = "\n\nclass Qwen2MLP(nn.Module):\n"
    start_marker = "# --- Codex constant steering patch start ---"
    end_marker = "# --- Codex constant steering patch end ---"
    already_patched = start_marker in text
    if already_patched:
        start = text.index(start_marker)
        end = text.index(end_marker, start) + len(end_marker)
        text = text[:start] + HELPER.strip() + text[end:]
    else:
        if marker not in text:
            raise ValueError("Could not find Qwen2MLP marker.")
        text = text.replace(marker, HELPER + marker, 1)

    if not already_patched:
        init_marker = "        super().__init__()\n        self.hidden_size = config.hidden_size\n"
        if init_marker not in text:
            raise ValueError("Could not find Qwen2DecoderLayer.__init__ marker.")
        text = text.replace(
            init_marker,
            "        super().__init__()\n        self.layer_idx = extract_layer_index(prefix)\n        self.hidden_size = config.hidden_size\n",
            1,
        )

    attn_marker = (
        "        hidden_states = self.self_attn(\n"
        "            positions=positions,\n"
        "            hidden_states=hidden_states,\n"
        "        )\n\n"
        "        # Fully Connected\n"
    )
    if not already_patched:
        if attn_marker not in text:
            raise ValueError("Could not find attention output marker.")
        text = text.replace(
            attn_marker,
            "        hidden_states = self.self_attn(\n"
            "            positions=positions,\n"
            "            hidden_states=hidden_states,\n"
            "        )\n"
            "        hidden_states = _qwen2_apply_constant_steering(hidden_states, self.layer_idx, \"post_attn\")\n\n"
            "        # Fully Connected\n",
            1,
        )

    mlp_marker = "        hidden_states = self.mlp(hidden_states)\n        return hidden_states, residual\n"
    if not already_patched:
        if mlp_marker not in text:
            raise ValueError("Could not find MLP output marker.")
        text = text.replace(
            mlp_marker,
            "        hidden_states = self.mlp(hidden_states)\n"
            "        hidden_states = _qwen2_apply_constant_steering(hidden_states, self.layer_idx, \"block_output\")\n"
            "        return hidden_states, residual\n",
            1,
        )

    path.write_text(text, encoding="utf-8")
    print(f"Patched: {path}")
    print(f"Backup: {backup}")


def patch_qwen3_file(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    backup = path.with_suffix(path.suffix + ".codex_bak")
    if not backup.exists():
        shutil.copy2(path, backup)

    if "from .qwen2 import _qwen2_apply_constant_steering" not in text:
        import_marker = "from .qwen2 import Qwen2Model\n"
        if import_marker not in text:
            raise ValueError("Could not find qwen3 Qwen2Model import marker.")
        text = text.replace(
            import_marker,
            import_marker + "from .qwen2 import _qwen2_apply_constant_steering\n",
            1,
        )

    if "self.layer_idx = extract_layer_index(prefix)" not in text:
        init_marker = "        super().__init__()\n        self.hidden_size = config.hidden_size\n"
        if init_marker not in text:
            raise ValueError("Could not find Qwen3DecoderLayer.__init__ marker.")
        text = text.replace(
            init_marker,
            "        super().__init__()\n        self.layer_idx = extract_layer_index(prefix)\n        self.hidden_size = config.hidden_size\n",
            1,
        )

    post_attn_call = '        hidden_states = _qwen2_apply_constant_steering(hidden_states, self.layer_idx, "post_attn")\n'
    if post_attn_call not in text:
        attn_marker = (
            "        hidden_states = self.self_attn(\n"
            "            positions=positions,\n"
            "            hidden_states=hidden_states,\n"
            "        )\n\n"
            "        # Fully Connected\n"
        )
        if attn_marker not in text:
            raise ValueError("Could not find Qwen3 attention output marker.")
        text = text.replace(
            attn_marker,
            "        hidden_states = self.self_attn(\n"
            "            positions=positions,\n"
            "            hidden_states=hidden_states,\n"
            "        )\n"
            + post_attn_call
            + "\n"
            "        # Fully Connected\n",
            1,
        )

    block_output_call = '        hidden_states = _qwen2_apply_constant_steering(hidden_states, self.layer_idx, "block_output")\n'
    if block_output_call not in text:
        mlp_marker = "        hidden_states = self.mlp(hidden_states)\n        return hidden_states, residual\n"
        if mlp_marker not in text:
            raise ValueError("Could not find Qwen3 MLP output marker.")
        text = text.replace(
            mlp_marker,
            "        hidden_states = self.mlp(hidden_states)\n"
            + block_output_call
            + "        return hidden_states, residual\n",
            1,
        )

    path.write_text(text, encoding="utf-8")
    print(f"Patched: {path}")
    print(f"Backup: {backup}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        default="",
        help="Path to vllm/model_executor/models/qwen2.py. If omitted, import vLLM and locate it.",
    )
    args = parser.parse_args()
    if args.path:
        path = Path(args.path)
    else:
        import vllm.model_executor.models.qwen2 as qwen2

        path = Path(qwen2.__file__)
    patch_qwen2_file(path)
    qwen3_path = path.with_name("qwen3.py")
    if qwen3_path.exists():
        patch_qwen3_file(qwen3_path)


if __name__ == "__main__":
    main()
