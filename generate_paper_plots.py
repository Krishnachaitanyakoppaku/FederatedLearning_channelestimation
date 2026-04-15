"""Generate paper-ready figures from structured experiment logs."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Generate mandatory figures for the paper")
    parser.add_argument("--raw_dir", type=str, default="results/raw", help="Raw run log directory")
    parser.add_argument("--summary_csv", type=str, default="results/summary/aggregated_metrics.csv", help="Aggregated summary CSV")
    parser.add_argument("--out_dir", type=str, default="results/figures", help="Figure output directory")
    parser.add_argument(
        "--manifest_path",
        type=str,
        default="results/figures/figure_manifest.csv",
        help="Figure manifest output path",
    )
    return parser.parse_args()


def load_runs(raw_dir):
    runs = []
    if not os.path.isdir(raw_dir):
        return runs
    for name in os.listdir(raw_dir):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(raw_dir, name), "r", encoding="utf-8") as f:
            runs.append(json.load(f))
    return runs


def load_summary_csv(path):
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        for line in f:
            vals = line.strip().split(",")
            if len(vals) != len(header):
                continue
            rows.append(dict(zip(header, vals)))
    return rows


def _as_float(value, default=np.nan):
    try:
        return float(value)
    except Exception:
        return default


def plot_loss_vs_epochs(runs, out_dir):
    loss_by_model = defaultdict(lambda: defaultdict(list))
    for run in runs:
        meta = run.get("metadata", {})
        if meta.get("setting") != "Centralized":
            continue
        model = meta.get("model", "Unknown")
        for point in run.get("history", []):
            if point.get("step_kind") != "epoch":
                continue
            epoch = int(point.get("step_index", 0))
            loss = _as_float(point.get("train_mse_loss"))
            if np.isfinite(loss):
                loss_by_model[model][epoch].append(loss)

    if not loss_by_model:
        return None

    plt.figure(figsize=(8, 5))
    for model, by_epoch in sorted(loss_by_model.items()):
        epochs = sorted(by_epoch.keys())
        means = [np.mean(by_epoch[e]) for e in epochs]
        plt.plot(epochs, means, marker="o", label=model)

    plt.xlabel("Epochs")
    plt.ylabel("MSE Loss")
    plt.title("Loss vs Epochs (Centralized)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(out_dir, "loss_vs_epochs.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def plot_nmse_vs_rounds(runs, out_dir):
    round_curves = defaultdict(lambda: defaultdict(list))
    for run in runs:
        meta = run.get("metadata", {})
        setting = meta.get("setting")
        if setting not in {"FedAvg", "FedProx"}:
            continue
        if meta.get("data_mode") != "Non-IID":
            continue
        for point in run.get("history", []):
            if point.get("step_kind") != "round":
                continue
            rnd = int(point.get("step_index", 0))
            nmse_db = _as_float(point.get("nmse_db"))
            if np.isfinite(nmse_db):
                round_curves[setting][rnd].append(nmse_db)

    if not round_curves:
        return None

    plt.figure(figsize=(8, 5))
    for setting in ["FedAvg", "FedProx"]:
        by_round = round_curves.get(setting)
        if not by_round:
            continue
        rounds = sorted(by_round.keys())
        means = [np.mean(by_round[r]) for r in rounds]
        plt.plot(rounds, means, marker="o", label=setting)

    plt.xlabel("Rounds")
    plt.ylabel("NMSE (dB)")
    plt.title("NMSE vs Rounds (Non-IID)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(out_dir, "nmse_vs_rounds.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def plot_nmse_model_bar(summary_rows, out_dir):
    if not summary_rows:
        return None

    # Keep one best row per model/setting/data_mode
    best = {}
    for row in summary_rows:
        key = (row["model"], row["setting"], row["data_mode"])
        nmse_db = _as_float(row["nmse_db_mean"])
        if key not in best or nmse_db < best[key]["nmse_db"]:
            best[key] = {"nmse_db": nmse_db}

    labels = []
    values = []
    for key, value in sorted(best.items()):
        model, setting, data_mode = key
        labels.append(f"{model}\n{setting}\n{data_mode}")
        values.append(value["nmse_db"])

    if not values:
        return None

    plt.figure(figsize=(10, 5))
    plt.bar(range(len(values)), values)
    plt.xticks(range(len(values)), labels, rotation=25, ha="right")
    plt.ylabel("NMSE (dB)")
    plt.title("NMSE vs Models/Settings")
    plt.tight_layout()
    out_path = os.path.join(out_dir, "nmse_vs_models_bar.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def plot_iid_vs_noniid(summary_rows, out_dir):
    if not summary_rows:
        return None

    pairs = defaultdict(dict)
    for row in summary_rows:
        if row.get("setting") != "FedAvg":
            continue
        model = row.get("model")
        mode = row.get("data_mode")
        nmse_db = _as_float(row.get("nmse_db_mean"))
        if not np.isfinite(nmse_db):
            continue
        current = pairs[model].get(mode)
        if current is None or nmse_db < current:
            pairs[model][mode] = nmse_db

    labels = []
    iid_vals = []
    non_iid_vals = []
    for model, modes in sorted(pairs.items()):
        if "IID" in modes and "Non-IID" in modes:
            labels.append(model)
            iid_vals.append(modes["IID"])
            non_iid_vals.append(modes["Non-IID"])

    if not labels:
        return None

    x = np.arange(len(labels))
    w = 0.35
    plt.figure(figsize=(8, 5))
    plt.bar(x - w / 2, iid_vals, w, label="IID")
    plt.bar(x + w / 2, non_iid_vals, w, label="Non-IID")
    plt.xticks(x, labels)
    plt.ylabel("NMSE (dB)")
    plt.title("IID vs Non-IID Comparison (FedAvg)")
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(out_dir, "iid_vs_noniid_bar.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def write_manifest(manifest_path, figure_sources):
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["figure_path", "source_kind", "source_reference"])
        writer.writeheader()
        for row in figure_sources:
            writer.writerow(row)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    runs = load_runs(args.raw_dir)
    summary_rows = load_summary_csv(args.summary_csv)

    run_sources = [run.get("metadata", {}).get("run_id", "unknown") for run in runs]
    figure_sources = []

    fig1 = plot_loss_vs_epochs(runs, args.out_dir)
    if fig1:
        for run_id in run_sources:
            figure_sources.append({"figure_path": fig1, "source_kind": "run_id", "source_reference": run_id})

    fig2 = plot_nmse_vs_rounds(runs, args.out_dir)
    if fig2:
        for run_id in run_sources:
            figure_sources.append({"figure_path": fig2, "source_kind": "run_id", "source_reference": run_id})

    fig3 = plot_nmse_model_bar(summary_rows, args.out_dir)
    if fig3:
        figure_sources.append(
            {
                "figure_path": fig3,
                "source_kind": "summary_csv",
                "source_reference": args.summary_csv,
            }
        )

    fig4 = plot_iid_vs_noniid(summary_rows, args.out_dir)
    if fig4:
        figure_sources.append(
            {
                "figure_path": fig4,
                "source_kind": "summary_csv",
                "source_reference": args.summary_csv,
            }
        )

    write_manifest(args.manifest_path, figure_sources)

    print(f"Figures saved to {args.out_dir}")
    print(f"Figure manifest saved to {args.manifest_path}")


if __name__ == "__main__":
    main()
