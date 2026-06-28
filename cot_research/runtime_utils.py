from __future__ import annotations

from typing import Any, List, Sequence, TypeVar

import torch


T = TypeVar("T")


def parse_parallel_gpu_ids(parallel_gpu_ids: str, *, default_to_visible: bool = True) -> List[int]:
    if parallel_gpu_ids.strip():
        out: List[int] = []
        seen = set()
        for token in parallel_gpu_ids.split(","):
            token = token.strip()
            if not token:
                continue
            gpu_id = int(token)
            if gpu_id < 0:
                raise ValueError(f"Invalid GPU id: {gpu_id}")
            if gpu_id not in seen:
                seen.add(gpu_id)
                out.append(gpu_id)
        if not out:
            raise ValueError("--parallel_gpu_ids is set but empty after parsing.")
        return out
    if default_to_visible and torch.cuda.is_available():
        return list(range(torch.cuda.device_count()))
    return []


def split_examples_contiguous(rows: Sequence[T], num_shards: int) -> List[List[T]]:
    if num_shards <= 0 or not rows:
        return []
    shard_size = (len(rows) + num_shards - 1) // num_shards
    shards = [list(rows[idx : idx + shard_size]) for idx in range(0, len(rows), shard_size)]
    return [shard for shard in shards if shard]


def resolve_device_map(device_map: Any, gpu_id: int) -> Any:
    if gpu_id >= 0:
        return {"": gpu_id}
    return device_map


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
