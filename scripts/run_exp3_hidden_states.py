#!/usr/bin/env python3
"""Experiment 3: Hidden state extraction for "certain" vs "hesitant" cases.

Collects two classes of reasoning traces:
  - Certain: model gives correct answer directly, short trace, no reflection
  - Hesitant: model reflects multiple times, long trace

Extracts residual stream hidden states at 10%, 30%, 50%, and pre-answer
positions. Produces PCA and UMAP 2D visualizations.

Usage:
  python scripts/run_exp3_hidden_states.py \
    --model Qwen/Qwen3-4B \
    --output-dir outputs/exp3_hidden_states \
    --gpu-ids 0 \
    --exp1-traces outputs/exp1_reasoning_trace/full_traces.jsonl \
    --exp1-analysis outputs/exp1_reasoning_trace/raw_traces.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."


def load_exp1_data(traces_path: str, analysis_path: str) -> List[Dict[str, Any]]:
    """Load traces and analysis from Experiment 1 output."""
    traces = {}
    with open(traces_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            key = (obj["problem_id"], obj["sample_idx"])
            traces[key] = obj["generated_text"]

    rows = []
    with open(analysis_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            key = (obj["problem_id"], obj["sample_idx"])
            obj["generated_text"] = traces.get(key, "")
            rows.append(obj)
    return rows


def classify_cases(
    rows: List[Dict[str, Any]],
    min_per_class: int = 15,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split into 'certain' and 'hesitant' cases."""
    certain = []
    hesitant = []

    for r in rows:
        if not r.get("generated_text"):
            continue
        if r.get("final_correct") and not r.get("has_reflection") and r.get("think_length", 9999) < 1500:
            certain.append(r)
        elif r.get("has_reflection") and r.get("reflection_count", 0) >= 2 and r.get("think_length", 0) > 500:
            hesitant.append(r)

    print(f"Found {len(certain)} certain cases, {len(hesitant)} hesitant cases")

    if len(certain) > min_per_class * 2:
        certain = certain[:min_per_class * 2]
    if len(hesitant) > min_per_class * 2:
        hesitant = hesitant[:min_per_class * 2]

    return certain, hesitant


def extract_hidden_states_at_positions(
    model,
    tokenizer,
    text: str,
    question: str,
    layer_idx: int = -2,
    positions_pct: List[float] = [0.1, 0.3, 0.5, 0.95],
) -> Dict[str, np.ndarray]:
    """Extract hidden states at specified percentage positions of the generated text."""
    import torch

    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    try:
        prompt = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True,
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
        )

    full_text = prompt + text
    inputs = tokenizer(full_text, return_tensors="pt", add_special_tokens=False, truncation=True, max_length=4096)
    input_ids = inputs["input_ids"].to(model.device)

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    prompt_len = len(prompt_ids)
    gen_len = input_ids.shape[-1] - prompt_len

    if gen_len <= 0:
        return {}

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            output_hidden_states=True,
            return_dict=True,
        )

    hidden_states = outputs.hidden_states
    num_layers = len(hidden_states)
    target_layer = layer_idx if layer_idx >= 0 else num_layers + layer_idx
    target_layer = max(0, min(target_layer, num_layers - 1))

    layer_hidden = hidden_states[target_layer][0]  # (seq_len, hidden_dim)

    results = {}
    for pct in positions_pct:
        pos = prompt_len + int(gen_len * pct)
        pos = min(pos, input_ids.shape[-1] - 1)
        h = layer_hidden[pos].detach().cpu().float().numpy()
        results[f"pct_{int(pct * 100)}"] = h

    return results


def run_hidden_state_collection(
    model,
    tokenizer,
    cases: List[Dict[str, Any]],
    label: str,
    layer_idx: int,
) -> List[Dict[str, Any]]:
    """Collect hidden states for a set of cases."""
    collected = []
    for i, case in enumerate(cases):
        text = case.get("generated_text", "")
        question = case.get("question", "")
        if not text or not question:
            continue

        print(f"  [{label}] {i+1}/{len(cases)}: {case['problem_id']}#{case['sample_idx']}")
        states = extract_hidden_states_at_positions(
            model, tokenizer, text, question, layer_idx=layer_idx,
        )
        if not states:
            continue

        collected.append({
            "problem_id": case["problem_id"],
            "sample_idx": case["sample_idx"],
            "label": label,
            "difficulty": case.get("difficulty", ""),
            "reflection_count": case.get("reflection_count", 0),
            "think_length": case.get("think_length", 0),
            "final_correct": case.get("final_correct", False),
            "hidden_states": {k: v.tolist() for k, v in states.items()},
        })
    return collected


def do_pca_umap(collected: List[Dict[str, Any]], output_dir: Path):
    """Perform PCA and UMAP, save plots and data."""
    from sklearn.decomposition import PCA

    try:
        import umap
        has_umap = True
    except ImportError:
        has_umap = False
        print("UMAP not available, skipping UMAP visualization")

    pct_keys = ["pct_10", "pct_30", "pct_50", "pct_95"]

    # build flat arrays
    vectors = []
    meta = []
    for item in collected:
        for pct_key in pct_keys:
            if pct_key in item["hidden_states"]:
                vectors.append(item["hidden_states"][pct_key])
                meta.append({
                    "problem_id": item["problem_id"],
                    "sample_idx": item["sample_idx"],
                    "label": item["label"],
                    "pct": pct_key,
                    "reflection_count": item["reflection_count"],
                    "think_length": item["think_length"],
                })

    if not vectors:
        print("No vectors to visualize")
        return

    X = np.array(vectors)
    print(f"PCA/UMAP input: {X.shape}")

    # PCA
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)

    pca_data = []
    for i, m in enumerate(meta):
        pca_data.append({
            **m,
            "pca_x": float(X_pca[i, 0]),
            "pca_y": float(X_pca[i, 1]),
            "pca_z": float(X_pca[i, 2]),
        })

    with open(output_dir / "pca_coordinates.json", "w") as f:
        json.dump(pca_data, f, indent=2)

    explained = pca.explained_variance_ratio_
    print(f"PCA explained variance: {explained[0]:.3f}, {explained[1]:.3f}, {explained[2]:.3f}")

    # generate matplotlib plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # PCA colored by label
        ax = axes[0]
        for label, color, marker in [("certain", "blue", "o"), ("hesitant", "red", "^")]:
            subset = [d for d in pca_data if d["label"] == label]
            for pct_key, alpha in [("pct_10", 0.3), ("pct_30", 0.5), ("pct_50", 0.7), ("pct_95", 1.0)]:
                pts = [d for d in subset if d["pct"] == pct_key]
                if pts:
                    ax.scatter(
                        [d["pca_x"] for d in pts],
                        [d["pca_y"] for d in pts],
                        c=color, marker=marker, alpha=alpha,
                        label=f"{label}_{pct_key}" if pct_key == "pct_95" else "",
                        s=40,
                    )
        ax.set_xlabel(f"PC1 ({explained[0]:.1%})")
        ax.set_ylabel(f"PC2 ({explained[1]:.1%})")
        ax.set_title("PCA: Certain vs Hesitant")
        ax.legend(fontsize=8)

        # PCA colored by time position
        ax = axes[1]
        pct_colors = {"pct_10": "green", "pct_30": "yellow", "pct_50": "orange", "pct_95": "red"}
        for pct_key, color in pct_colors.items():
            pts = [d for d in pca_data if d["pct"] == pct_key]
            certain_pts = [d for d in pts if d["label"] == "certain"]
            hesitant_pts = [d for d in pts if d["label"] == "hesitant"]
            if certain_pts:
                ax.scatter(
                    [d["pca_x"] for d in certain_pts],
                    [d["pca_y"] for d in certain_pts],
                    c=color, marker="o", alpha=0.6, s=30,
                )
            if hesitant_pts:
                ax.scatter(
                    [d["pca_x"] for d in hesitant_pts],
                    [d["pca_y"] for d in hesitant_pts],
                    c=color, marker="^", alpha=0.6, s=30,
                )
        ax.set_xlabel(f"PC1 ({explained[0]:.1%})")
        ax.set_ylabel(f"PC2 ({explained[1]:.1%})")
        ax.set_title("PCA: Colored by Time Position\n(green=10% → red=95%, o=certain ^=hesitant)")

        plt.tight_layout()
        plt.savefig(output_dir / "pca_visualization.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved PCA plot to {output_dir / 'pca_visualization.png'}")

        # trajectory plot
        fig, ax = plt.subplots(figsize=(10, 8))
        for label, color in [("certain", "blue"), ("hesitant", "red")]:
            cases_subset = [c for c in collected if c["label"] == label]
            for case in cases_subset[:8]:
                xs, ys = [], []
                for pct_key in pct_keys:
                    pts = [d for d in pca_data
                           if d["problem_id"] == case["problem_id"]
                           and d["sample_idx"] == case["sample_idx"]
                           and d["pct"] == pct_key]
                    if pts:
                        xs.append(pts[0]["pca_x"])
                        ys.append(pts[0]["pca_y"])
                if len(xs) >= 2:
                    ax.plot(xs, ys, c=color, alpha=0.4, linewidth=1)
                    ax.scatter(xs[0], ys[0], c=color, marker="s", s=50, zorder=5)
                    ax.scatter(xs[-1], ys[-1], c=color, marker="*", s=80, zorder=5)

        ax.set_xlabel(f"PC1 ({explained[0]:.1%})")
        ax.set_ylabel(f"PC2 ({explained[1]:.1%})")
        ax.set_title("Hidden State Trajectories\n(square=start, star=end, blue=certain, red=hesitant)")
        plt.savefig(output_dir / "trajectory_visualization.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved trajectory plot")

    except ImportError:
        print("matplotlib not available, skipping plot generation")

    # UMAP
    if has_umap:
        try:
            reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
            X_umap = reducer.fit_transform(X)

            umap_data = []
            for i, m in enumerate(meta):
                umap_data.append({
                    **m,
                    "umap_x": float(X_umap[i, 0]),
                    "umap_y": float(X_umap[i, 1]),
                })

            with open(output_dir / "umap_coordinates.json", "w") as f:
                json.dump(umap_data, f, indent=2)

            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(10, 8))
                for label, color, marker in [("certain", "blue", "o"), ("hesitant", "red", "^")]:
                    subset = [d for d in umap_data if d["label"] == label]
                    for pct_key, alpha in [("pct_10", 0.3), ("pct_30", 0.5), ("pct_50", 0.7), ("pct_95", 1.0)]:
                        pts = [d for d in subset if d["pct"] == pct_key]
                        if pts:
                            ax.scatter(
                                [d["umap_x"] for d in pts],
                                [d["umap_y"] for d in pts],
                                c=color, marker=marker, alpha=alpha, s=40,
                                label=f"{label}_{pct_key}" if pct_key == "pct_95" else "",
                            )
                ax.set_title("UMAP: Certain vs Hesitant")
                ax.legend(fontsize=8)
                plt.savefig(output_dir / "umap_visualization.png", dpi=150, bbox_inches="tight")
                plt.close()
                print(f"Saved UMAP plot")
            except ImportError:
                pass
        except Exception as e:
            print(f"UMAP failed: {e}")


def find_anomalous_cases(
    collected: List[Dict[str, Any]],
    pca_data: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Find cases whose PCA position is anomalous for their class."""
    if not pca_data:
        return []

    certain_pts = [d for d in pca_data if d["label"] == "certain" and d["pct"] == "pct_95"]
    hesitant_pts = [d for d in pca_data if d["label"] == "hesitant" and d["pct"] == "pct_95"]

    if not certain_pts or not hesitant_pts:
        return []

    certain_cx = np.mean([d["pca_x"] for d in certain_pts])
    certain_cy = np.mean([d["pca_y"] for d in certain_pts])
    hesitant_cx = np.mean([d["pca_x"] for d in hesitant_pts])
    hesitant_cy = np.mean([d["pca_y"] for d in hesitant_pts])

    anomalous = []
    for d in pca_data:
        if d["pct"] != "pct_95":
            continue
        px, py = d["pca_x"], d["pca_y"]
        dist_own = np.sqrt((px - (certain_cx if d["label"] == "certain" else hesitant_cx))**2 +
                          (py - (certain_cy if d["label"] == "certain" else hesitant_cy))**2)
        dist_other = np.sqrt((px - (hesitant_cx if d["label"] == "certain" else certain_cx))**2 +
                            (py - (hesitant_cy if d["label"] == "certain" else certain_cy))**2)
        if dist_other < dist_own:
            anomalous.append(d)

    return anomalous[:5]


def generate_report(
    certain: List[Dict[str, Any]],
    hesitant: List[Dict[str, Any]],
    anomalous: List[Dict[str, Any]],
    output_dir: Path,
) -> str:
    lines = ["# Experiment 3: Hidden State Analysis Report\n"]
    lines.append(f"## Dataset")
    lines.append(f"- Certain cases: {len(certain)}")
    lines.append(f"- Hesitant cases: {len(hesitant)}")
    lines.append(f"- Anomalous cases found: {len(anomalous)}\n")

    lines.append("## Visualizations")
    lines.append("- PCA: see `pca_visualization.png`")
    lines.append("- Trajectories: see `trajectory_visualization.png`")
    lines.append("- UMAP: see `umap_visualization.png` (if available)\n")

    lines.append("## Certain Cases (samples)")
    for c in certain[:5]:
        lines.append(f"\n### {c['problem_id']}#{c['sample_idx']}")
        lines.append(f"- Think length: {c.get('think_length', '?')}, Reflections: {c.get('reflection_count', 0)}")
        text = c.get("generated_text", "")[:1000]
        lines.append(f"<details><summary>Trace</summary>\n\n```\n{text}\n```\n</details>")

    lines.append("\n## Hesitant Cases (samples)")
    for c in hesitant[:5]:
        lines.append(f"\n### {c['problem_id']}#{c['sample_idx']}")
        lines.append(f"- Think length: {c.get('think_length', '?')}, Reflections: {c.get('reflection_count', 0)}")
        text = c.get("generated_text", "")[:1500]
        lines.append(f"<details><summary>Trace</summary>\n\n```\n{text}\n```\n</details>")

    if anomalous:
        lines.append("\n## Anomalous Cases (position mismatches)")
        for a in anomalous:
            lines.append(f"\n### {a['problem_id']}#{a['sample_idx']}")
            lines.append(f"- Label: {a['label']} but positioned near the other cluster")
            lines.append(f"- PCA: ({a['pca_x']:.3f}, {a['pca_y']:.3f})")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Exp3: Hidden state extraction and visualization")
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--output-dir", default="outputs/exp3_hidden_states")
    parser.add_argument("--gpu-ids", default="0")
    parser.add_argument("--layer", type=int, default=-2, help="Layer index for hidden state extraction")
    parser.add_argument("--max-cases", type=int, default=20, help="Max cases per class")
    parser.add_argument("--exp1-traces", default="outputs/exp1_reasoning_trace/full_traces.jsonl")
    parser.add_argument("--exp1-analysis", default="outputs/exp1_reasoning_trace/raw_traces.jsonl")
    args = parser.parse_args()

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # load exp1 data
    print("Loading Experiment 1 data...")
    rows = load_exp1_data(args.exp1_traces, args.exp1_analysis)
    print(f"Loaded {len(rows)} traces")

    certain, hesitant = classify_cases(rows, min_per_class=args.max_cases)
    certain = certain[:args.max_cases]
    hesitant = hesitant[:args.max_cases]

    if len(certain) < 5 or len(hesitant) < 5:
        print(f"WARNING: Not enough cases (certain={len(certain)}, hesitant={len(hesitant)})")
        print("Relaxing criteria...")
        certain_relaxed = [r for r in rows if r.get("final_correct") and r.get("reflection_count", 0) <= 1]
        hesitant_relaxed = [r for r in rows if r.get("reflection_count", 0) >= 1 and r.get("think_length", 0) > 300]
        certain = (certain + certain_relaxed)[:args.max_cases]
        hesitant = (hesitant + hesitant_relaxed)[:args.max_cases]

    print(f"Using {len(certain)} certain, {len(hesitant)} hesitant cases")

    # load model
    print(f"Loading model {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True,
    )
    model.eval()

    # extract hidden states
    print("Extracting hidden states for certain cases...")
    certain_collected = run_hidden_state_collection(model, tokenizer, certain, "certain", args.layer)
    print("Extracting hidden states for hesitant cases...")
    hesitant_collected = run_hidden_state_collection(model, tokenizer, hesitant, "hesitant", args.layer)

    all_collected = certain_collected + hesitant_collected
    print(f"Total collected: {len(all_collected)}")

    # save raw hidden states
    with open(output_dir / "hidden_states_meta.json", "w") as f:
        json.dump([{k: v for k, v in c.items() if k != "hidden_states"} for c in all_collected], f, indent=2)

    # PCA/UMAP
    print("Running PCA/UMAP...")
    do_pca_umap(all_collected, output_dir)

    # find anomalous
    pca_path = output_dir / "pca_coordinates.json"
    anomalous = []
    if pca_path.exists():
        with open(pca_path) as f:
            pca_data = json.load(f)
        anomalous = find_anomalous_cases(all_collected, pca_data)

    # generate report
    report = generate_report(certain, hesitant, anomalous, output_dir)
    (output_dir / "hidden_state_report.md").write_text(report, encoding="utf-8")

    with open(output_dir / "run_config.json", "w") as f:
        json.dump({
            "model": args.model, "layer": args.layer,
            "certain_count": len(certain_collected),
            "hesitant_count": len(hesitant_collected),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, f, indent=2)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
