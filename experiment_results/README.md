# Experiment Results Archive

This folder consolidates experiment artifacts that were previously scattered across the repository.

Structure:
- `local_current/`: result artifacts collected from local `outputs*` directories.
- `remote_server/outputs/`: result artifacts downloaded from the remote server via `rsync`.
- `reports/`: local experiment reports and summaries.

Filtering rules:
- Raw CoT JSON/JSONL data were intentionally excluded.
- Large generation shards such as `numinamath_shards/*.jsonl` were excluded.
- Per-example raw generation files such as `rows.jsonl`, `_worker_*_rows.jsonl`, `_worker_*_ex.jsonl`, `all_repetition_cases.jsonl`, `*.repetition_cases.jsonl`, and `ablation_*_CoT.jsonl` were excluded.
- Summary-oriented artifacts such as `csv`, `json`, `md`, configs, and metric JSONL files were kept.

Rsync rule files used for this sync are stored in `rsync/upload_excludes.txt` and `rsync/results_excludes.txt`.
