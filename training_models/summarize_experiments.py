"""Consolidates key metrics from multiple main.py runs into a single CSV +
printed table, one row per experiment.

Usage:
    python summarize_experiments.py results/cifar10/*/seed42/mobilenet_v2 --out results/cifar10_summary.csv

Each positional arg is a save_path prefix (same value passed to main.py's
--save_path). Two output formats are supported, since not every training
mode in this codebase has been migrated to MetricsRecorder:
  - MetricsRecorder-based (baseline, train_with_revision, critical_periods_dbpd,
    soft_dbpd, delayed_dbpd, combined): reads "{prefix}_run_metadata.json" and
    "{prefix}_epochs.parquet".
  - Legacy JSON (train_with_random/SMRD, train_with_percentage, etc., which
    predate MetricsRecorder and never write a "_run_metadata.json"): falls
    back to "{prefix}_test", a JSON file with per-epoch "accuracy" and
    "cumulative_time" arrays. condition/seed are inferred from the save_path
    itself (assumes the results/{dataset}/{condition}/seed{n}/{model}
    convention used by launch_all.sh), since the legacy format doesn't
    record them.
"""
import argparse
import json
import os

import pandas as pd


def _infer_condition_and_seed(save_path_prefix):
    parts = save_path_prefix.replace("\\", "/").rstrip("/").split("/")
    condition = parts[-3] if len(parts) >= 3 else None
    seed_part = parts[-2] if len(parts) >= 2 else None
    seed = seed_part.replace("seed", "") if seed_part and seed_part.startswith("seed") else seed_part
    return condition, seed


def summarize_metrics_recorder_run(save_path_prefix):
    metadata_path = f"{save_path_prefix}_run_metadata.json"
    epochs_path = f"{save_path_prefix}_epochs.parquet"

    with open(metadata_path) as f:
        metadata = json.load(f)

    df = pd.read_parquet(epochs_path)
    if df.empty:
        raise ValueError(f"{epochs_path} has no epoch rows")

    best_row = df.loc[df["val_accuracy"].idxmax()]
    final_row = df.iloc[-1]

    # samples_total already equals the number of forward passes performed
    # (every sample in the loader is forwarded each epoch to decide the
    # hard/easy mask, before any masking); samples_used is the smaller,
    # masked-down backward-pass count. See MetricsRecorder's docstring.
    total_samples_forward = int(df["samples_total"].sum())
    total_samples_backward = int(df["samples_used"].sum())
    total_time_seconds = float(df["time_seconds"].sum())

    return {
        "condition": metadata.get("mode"),
        "save_path": save_path_prefix,
        "seed": metadata.get("seed"),
        "threshold": metadata.get("threshold"),
        "onset": metadata.get("onset"),
        "keep_rate": metadata.get("keep_rate"),
        "critical_window": metadata.get("critical_window"),
        "critical_period_end_epoch": metadata.get("critical_period_end_epoch"),
        "epochs_completed": len(df),
        "best_val_accuracy": best_row["val_accuracy"],
        "best_val_epoch": int(best_row["epoch"]),
        "final_val_accuracy": final_row["val_accuracy"],
        "final_train_accuracy": final_row["train_accuracy"],
        "final_val_loss": final_row["val_loss"],
        "total_samples_forward": total_samples_forward,
        "total_samples_backward": total_samples_backward,
        "pct_samples_backward": round(100 * total_samples_backward / total_samples_forward, 1) if total_samples_forward else None,
        "total_time_seconds": total_time_seconds,
        "total_time_minutes": round(total_time_seconds / 60, 1),
        "format": "metrics_recorder",
    }


def summarize_legacy_run(save_path_prefix):
    test_path = f"{save_path_prefix}_test"
    with open(test_path) as f:
        test_data = json.load(f)

    key = next(iter(test_data))
    accuracy = test_data[key]["accuracy"]
    cumulative_time = test_data[key]["cumulative_time"]
    if not accuracy:
        raise ValueError(f"{test_path} has no accuracy entries")

    condition, seed = _infer_condition_and_seed(save_path_prefix)
    best_epoch = max(range(len(accuracy)), key=lambda i: accuracy[i])

    return {
        "condition": condition,
        "save_path": save_path_prefix,
        "seed": seed,
        "threshold": None,
        "onset": None,
        "keep_rate": None,
        "critical_window": None,
        "critical_period_end_epoch": None,
        "epochs_completed": len(accuracy),
        "best_val_accuracy": accuracy[best_epoch],
        "best_val_epoch": best_epoch,
        "final_val_accuracy": accuracy[-1],
        "final_train_accuracy": None,
        "final_val_loss": None,
        "total_samples_forward": None,
        "total_samples_backward": None,
        "pct_samples_backward": None,
        "total_time_seconds": cumulative_time[-1] if cumulative_time else None,
        "total_time_minutes": round(cumulative_time[-1] / 60, 1) if cumulative_time else None,
        "format": "legacy_json",
    }


def summarize_run(save_path_prefix):
    if os.path.exists(f"{save_path_prefix}_run_metadata.json"):
        return summarize_metrics_recorder_run(save_path_prefix)
    return summarize_legacy_run(save_path_prefix)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("save_paths", nargs="+", help="One or more save_path prefixes (as passed to main.py --save_path)")
    parser.add_argument("--out", default=None, help="Write the combined table to this CSV path")
    args = parser.parse_args()

    rows = []
    for save_path in args.save_paths:
        try:
            rows.append(summarize_run(save_path))
        except FileNotFoundError as e:
            print(f"[skip] {save_path}: {e}")

    if not rows:
        print("No runs found.")
        return

    summary = pd.DataFrame(rows).sort_values("condition")
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print(summary.to_string(index=False))

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True) if os.path.dirname(args.out) else None
        summary.to_csv(args.out, index=False)
        print(f"\nSaved combined log to {args.out}")


if __name__ == "__main__":
    main()
