from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import importlib
import sys

from .io_utils import dump_jsonl, load_jsonl
from .row_utils import build_example_id, build_problem_text, build_reference_solution
from .schemas import DatasetExample


REQUIRED_DATASET_FIELDS = ["id"]


def validate_dataset_records(records: Iterable[Dict[str, Any]]) -> None:
    for idx, row in enumerate(records):
        for key in REQUIRED_DATASET_FIELDS:
            if key not in row:
                raise ValueError(f"Dataset record index {idx} missing required field '{key}'.")
        if not str(row.get("question", "") or "").strip() and not str(row.get("prompt_prefix", "") or "").strip():
            raise ValueError(
                f"Dataset record index {idx} must provide either 'question' or 'prompt_prefix'."
            )


def load_examples(path: str) -> List[DatasetExample]:
    raw = load_jsonl(path)
    validate_dataset_records(raw)
    return [DatasetExample.from_dict(item) for item in raw]


def load_numinamath_dataset(
    *,
    dataset_name: str,
    dataset_split: str,
    cache_dir: str,
    local_files_only: bool,
    source_filter: str = "",
    shuffle: bool = False,
    seed: int = 1234,
    start_idx: int = 0,
    sample_stride: int = 1,
    max_examples: int = -1,
):
    try:
        DownloadConfig, Dataset, concatenate_datasets, load_dataset = _import_hf_datasets_symbols(
            "DownloadConfig",
            "Dataset",
            "concatenate_datasets",
            "load_dataset",
        )
    except ImportError as exc:
        raise RuntimeError(
            "This script requires the `datasets` package to load AI-MO/NuminaMath-CoT. "
            "Install it into the existing qwen_math environment, for example:\n"
            "  pip install datasets\n"
            "No new environment is required."
        ) from exc

    if local_files_only:
        cache_root = Path(cache_dir)
        direct_cached = _try_load_numinamath_from_arrow_cache(
            cache_root=cache_root,
            dataset_split=dataset_split,
            dataset_cls=Dataset,
            concat_fn=concatenate_datasets,
        )
        if direct_cached is not None:
            dataset = direct_cached
            if source_filter.strip():
                allowed = {item.strip() for item in source_filter.split(",") if item.strip()}
                dataset = dataset.filter(lambda row: str(row.get("source", "")) in allowed)
            if shuffle:
                dataset = dataset.shuffle(seed=seed)
            if start_idx > 0 or sample_stride > 1 or max_examples != -1:
                if sample_stride <= 0:
                    raise ValueError("--sample_stride must be >= 1")
                if max_examples > 0:
                    indices = [start_idx + i * sample_stride for i in range(max_examples)]
                else:
                    indices = list(range(start_idx, len(dataset), sample_stride))
                indices = [idx for idx in indices if idx < len(dataset)]
                dataset = dataset.select(indices)
            return dataset

    download_config = DownloadConfig(local_files_only=local_files_only)
    dataset = load_dataset(
        dataset_name,
        split=dataset_split,
        cache_dir=cache_dir,
        streaming=False,
        download_config=download_config,
    )
    if source_filter.strip():
        allowed = {item.strip() for item in source_filter.split(",") if item.strip()}
        dataset = dataset.filter(lambda row: str(row.get("source", "")) in allowed)
    if shuffle:
        dataset = dataset.shuffle(seed=seed)

    if start_idx > 0 or sample_stride > 1 or max_examples != -1:
        if sample_stride <= 0:
            raise ValueError("--sample_stride must be >= 1")
        if max_examples > 0:
            indices = [start_idx + i * sample_stride for i in range(max_examples)]
        else:
            indices = list(range(start_idx, len(dataset), sample_stride))
        indices = [idx for idx in indices if idx < len(dataset)]
        dataset = dataset.select(indices)
    return dataset


def _try_load_numinamath_from_arrow_cache(
    *,
    cache_root: Path,
    dataset_split: str,
    dataset_cls,
    concat_fn,
):
    if not cache_root.exists():
        return None

    split = str(dataset_split).strip().lower()
    if split not in {"train", "test"}:
        return None

    arrow_patterns = [
        f"**/*numina*{split}*.arrow",
        f"**/*Numina*{split}*.arrow",
    ]
    arrow_paths: List[Path] = []
    seen = set()
    for pattern in arrow_patterns:
        for path in sorted(cache_root.glob(pattern)):
            if not path.is_file():
                continue
            resolved = str(path.resolve())
            if resolved in seen:
                continue
            seen.add(resolved)
            arrow_paths.append(path)

    if not arrow_paths:
        return None

    shards = [dataset_cls.from_file(str(path)) for path in arrow_paths]
    if not shards:
        return None
    if len(shards) == 1:
        return shards[0]
    return concat_fn(shards)


def load_hf_dataset_records(
    *,
    dataset_name: str,
    dataset_split: str,
    cache_dir: str,
    local_files_only: bool,
    config_name: Optional[str] = None,
):
    try:
        DownloadConfig, load_dataset = _import_hf_datasets_symbols(
            "DownloadConfig",
            "load_dataset",
        )
    except ImportError as exc:
        raise RuntimeError(
            "This script requires the `datasets` package to load Hugging Face datasets. "
            "Install it into the existing qwen_math environment, for example:\n"
            "  pip install datasets\n"
            "No new environment is required."
        ) from exc

    download_config = DownloadConfig(local_files_only=local_files_only)
    kwargs = {
        "path": dataset_name,
        "split": dataset_split,
        "cache_dir": cache_dir,
        "streaming": False,
        "download_config": download_config,
    }
    if config_name:
        kwargs["name"] = config_name
    return load_dataset(**kwargs)


def _import_hf_datasets_symbols(*names: str):
    repo_root = Path(__file__).resolve().parents[1]
    blocked = {
        str(repo_root),
        "",
        ".",
    }
    original_sys_path = list(sys.path)
    original_module = sys.modules.pop("datasets", None)
    try:
        sys.path = [item for item in sys.path if item not in blocked]
        module = importlib.import_module("datasets")
        values = []
        for name in names:
            values.append(getattr(module, name))
        return tuple(values)
    finally:
        sys.path = original_sys_path
        if original_module is not None:
            sys.modules["datasets"] = original_module


def convert_numinamath_row(row: Dict[str, Any], dataset_local_index: int, *, dataset_name: str, dataset_split: str) -> Dict[str, Any]:
    return {
        "example_id": build_example_id(row, dataset_local_index),
        "id": row.get("id", row.get("idx", dataset_local_index)),
        "source": row.get("source"),
        "dataset_local_index": dataset_local_index,
        "problem": build_problem_text(row),
        "question": build_problem_text(row),
        "reference_solution": build_reference_solution(row),
        "reference_messages": row.get("messages"),
        "metadata": {
            "dataset_name": dataset_name,
            "dataset_split": dataset_split,
            "dataset_local_index": dataset_local_index,
            "source": row.get("source"),
        },
    }
