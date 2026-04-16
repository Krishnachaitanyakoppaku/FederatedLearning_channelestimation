"""Generate publication-grade metrics tables and comparison plots from run logs.

This script consolidates all post-processing needed for paper reporting:
- Run-level metrics extraction
- Main comparison and ablation tables
- Reliability and significance tables
- Complexity, latency, and communication-efficiency tables
- Comparison plots (SNR, rounds, ablations, Pareto, CDF, latency)
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy import stats  # type: ignore
except Exception:
    stats = None


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paper-grade reporting package")
    parser.add_argument("--raw_dir", type=str, default="results/raw", help="Run log directory")
    parser.add_argument("--summary_dir", type=str, default="results/summary", help="Summary table output dir")
    parser.add_argument("--figures_dir", type=str, default="results/figures/paper", help="Figure output dir")
    parser.add_argument("--manifest", type=str, default="results/figures/paper/figure_manifest.csv", help="Figure manifest CSV path")
    parser.add_argument("--dataset_dir", type=str, default="dataset/cnn/10dB", help="Dataset dir for latency benchmarking")
    parser.add_argument("--latency_repeats", type=int, default=50, help="Latency benchmark repeats")
    parser.add_argument("--latency_warmup", type=int, default=10, help="Latency benchmark warmup iterations")
    parser.add_argument("--default_clients", type=int, default=5, help="Fallback number of clients for FL communication estimates")
    parser.add_argument("--reliability_threshold_db", type=float, default=-18.0, help="Reliability threshold for success rate")
    return parser.parse_args()


def load_runs(raw_dir: str) -> List[dict]:
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        return []
    runs: List[dict] = []
    for p in sorted(raw_path.glob("*.json")):
        try:
            payload = pd.read_json(p, typ="series").to_dict()
        except Exception:
            continue
        if "metadata" in payload and "summary" in payload:
            runs.append(payload)
    return runs


def _safe_float(value, default=np.nan) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _history_best_nmse(history: Iterable[dict]) -> float:
    vals = [
        _safe_float(step.get("nmse_db"))
        for step in history
        if step.get("step_kind") == "round" and "nmse_db" in step
    ]
    vals = [v for v in vals if np.isfinite(v)]
    return min(vals) if vals else np.nan


def _history_rounds_to_threshold(history: Iterable[dict], threshold_db: float) -> float:
    for step in history:
        if step.get("step_kind") != "round":
            continue
        nmse_db = _safe_float(step.get("nmse_db"))
        if np.isfinite(nmse_db) and nmse_db <= threshold_db:
            return float(step.get("step_index", np.nan))
    return np.nan


def build_run_level_df(runs: List[dict], threshold_db: float) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for run in runs:
        meta = run.get("metadata", {})
        summary = run.get("summary", {})
        history = run.get("history", [])
        extra = meta.get("extra") or {}
        row: Dict[str, object] = {
            "run_id": meta.get("run_id"),
            "model": meta.get("model"),
            "setting": meta.get("setting"),
            "data_mode": meta.get("data_mode"),
            "snr_db": meta.get("snr_db"),
            "seed": meta.get("seed"),
            "local_epochs": meta.get("local_epochs"),
            "global_rounds": meta.get("global_rounds"),
            "mu": meta.get("mu"),
            "lr": meta.get("lr"),
            "batch_size": meta.get("batch_size"),
            "device": meta.get("device"),
            "git_commit": meta.get("git_commit"),
            "config_sha256": meta.get("config_sha256"),
            "num_clients": extra.get("num_clients"),
            "duration_sec": _safe_float(run.get("duration_sec")),
            "final_nmse_db": _safe_float(summary.get("final_nmse_db")),
            "final_nmse_linear": _safe_float(summary.get("final_nmse_linear")),
            "best_round_nmse_db": _history_best_nmse(history),
            "rounds_to_threshold": _history_rounds_to_threshold(history, threshold_db),
        }
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _ci95(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce").dropna().to_numpy()
    if len(vals) <= 1:
        return 0.0
    return float(1.96 * np.std(vals, ddof=0) / math.sqrt(len(vals)))


def build_main_comparison_table(run_df: pd.DataFrame) -> pd.DataFrame:
    if run_df.empty:
        return pd.DataFrame()

    grouped = (
        run_df.groupby([
            "model",
            "setting",
            "data_mode",
            "snr_db",
            "local_epochs",
            "global_rounds",
            "mu",
        ], dropna=False)["final_nmse_db"]
        .agg(["count", "mean", "std"])
        .reset_index()
    )
    grouped["ci95"] = (
        run_df.groupby([
            "model",
            "setting",
            "data_mode",
            "snr_db",
            "local_epochs",
            "global_rounds",
            "mu",
        ], dropna=False)["final_nmse_db"]
        .apply(_ci95)
        .to_numpy()
    )

    grouped = grouped.rename(
        columns={
            "count": "runs",
            "mean": "nmse_db_mean",
            "std": "nmse_db_std",
        }
    )
    grouped["nmse_db_std"] = grouped["nmse_db_std"].fillna(0.0)

    central = (
        grouped[grouped["setting"] == "Centralized"]
        .set_index(["model", "snr_db"])["nmse_db_mean"]
        .to_dict()
    )
    grouped["centralized_gap_db"] = grouped.apply(
        lambda r: r["nmse_db_mean"] - central.get((r["model"], r["snr_db"]), np.nan),
        axis=1,
    )
    return grouped.sort_values(["model", "setting", "data_mode", "snr_db"])


def build_ablation_table(main_df: pd.DataFrame) -> pd.DataFrame:
    if main_df.empty:
        return pd.DataFrame()
    ablation = main_df[
        main_df["setting"].isin(["FedAvg", "FedProx"])
    ][[
        "model",
        "setting",
        "data_mode",
        "snr_db",
        "global_rounds",
        "local_epochs",
        "mu",
        "nmse_db_mean",
        "nmse_db_std",
        "ci95",
        "runs",
    ]].copy()
    return ablation.sort_values(["model", "setting", "data_mode", "global_rounds", "local_epochs", "mu"])


def build_significance_table(run_df: pd.DataFrame) -> pd.DataFrame:
    if run_df.empty:
        return pd.DataFrame()

    rows: List[Dict[str, object]] = []
    keys = ["model", "snr_db"]
    for _, block in run_df.groupby(keys, dropna=False):
        labels = (
            block[["setting", "data_mode"]]
            .drop_duplicates()
            .apply(lambda r: f"{r['setting']}|{r['data_mode']}", axis=1)
            .tolist()
        )
        labels = sorted(labels)
        for a, b in combinations(labels, 2):
            set_a, mode_a = a.split("|", 1)
            set_b, mode_b = b.split("|", 1)
            vals_a = block[(block["setting"] == set_a) & (block["data_mode"] == mode_a)]["final_nmse_db"].dropna().to_numpy()
            vals_b = block[(block["setting"] == set_b) & (block["data_mode"] == mode_b)]["final_nmse_db"].dropna().to_numpy()
            if len(vals_a) < 2 or len(vals_b) < 2:
                continue
            if stats is not None:
                _, t_p = stats.ttest_ind(vals_a, vals_b, equal_var=False)
                _, u_p = stats.mannwhitneyu(vals_a, vals_b, alternative="two-sided")
            else:
                t_p = np.nan
                u_p = np.nan
            rows.append(
                {
                    "model": block.iloc[0]["model"],
                    "snr_db": block.iloc[0]["snr_db"],
                    "group_a": a,
                    "group_b": b,
                    "n_a": len(vals_a),
                    "n_b": len(vals_b),
                    "mean_a_nmse_db": float(np.mean(vals_a)),
                    "mean_b_nmse_db": float(np.mean(vals_b)),
                    "delta_a_minus_b_db": float(np.mean(vals_a) - np.mean(vals_b)),
                    "ttest_pvalue": float(t_p),
                    "mannwhitney_pvalue": float(u_p),
                    "ttest_significant_0_05": bool(np.isfinite(t_p) and t_p < 0.05),
                    "mannwhitney_significant_0_05": bool(np.isfinite(u_p) and u_p < 0.05),
                }
            )
    return pd.DataFrame(rows)


def _load_model_param_counts() -> Dict[str, int]:
    try:
        sys.path.append(str(PROJECT_ROOT / "Centralised Learning" / "CNN"))
        sys.path.append(str(PROJECT_ROOT / "Centralised Learning" / "ResUNet"))
        from cnn_model import ChannelEstimatorCNN, count_parameters  # type: ignore
        from resunet_model import ResUNet  # type: ignore

        cnn_params = count_parameters(ChannelEstimatorCNN())
        resunet_params = int(sum(p.numel() for p in ResUNet().parameters() if p.requires_grad))
        return {"CNN": cnn_params, "ResUNet": resunet_params}
    except Exception:
        return {}


def _detect_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    except Exception:
        return "cpu"


def benchmark_latency_ms_per_sample(dataset_dir: str, repeats: int, warmup: int) -> pd.DataFrame:
    try:
        import torch
    except Exception:
        return pd.DataFrame()

    sys.path.append(str(PROJECT_ROOT / "Centralised Learning" / "CNN"))
    sys.path.append(str(PROJECT_ROOT / "Centralised Learning" / "ResUNet"))
    from cnn_model import ChannelEstimatorCNN  # type: ignore
    from resunet_model import ResUNet  # type: ignore

    x_test = np.load(Path(dataset_dir) / "X_test.npy")
    x = torch.from_numpy(x_test[:64]).float()
    device_str = _detect_device()
    device = torch.device(device_str)
    x = x.to(device)

    records: List[Dict[str, object]] = []
    models = [("CNN", ChannelEstimatorCNN()), ("ResUNet", ResUNet())]
    for model_name, model in models:
        model = model.to(device)
        model.eval()
        with torch.no_grad():
            for _ in range(max(0, warmup)):
                _ = model(x)
            if device_str == "cuda":
                torch.cuda.synchronize()
            elif device_str == "mps":
                torch.mps.synchronize()

            durations = []
            for _ in range(max(1, repeats)):
                t0 = time.perf_counter()
                _ = model(x)
                if device_str == "cuda":
                    torch.cuda.synchronize()
                elif device_str == "mps":
                    torch.mps.synchronize()
                t1 = time.perf_counter()
                durations.append((t1 - t0) * 1000.0 / x.shape[0])

        records.append(
            {
                "model": model_name,
                "device": device_str,
                "latency_ms_per_sample_mean": float(np.mean(durations)),
                "latency_ms_per_sample_std": float(np.std(durations)),
            }
        )
    return pd.DataFrame(records)


def build_complexity_table(run_df: pd.DataFrame, latency_df: pd.DataFrame) -> pd.DataFrame:
    if run_df.empty:
        return pd.DataFrame()
    params = _load_model_param_counts()
    model_runtime = run_df.groupby("model", dropna=False)["duration_sec"].agg(["count", "mean"]).reset_index()
    model_runtime = model_runtime.rename(columns={"count": "runs", "mean": "duration_sec_mean"})
    model_runtime["params"] = model_runtime["model"].map(params)
    model_runtime["model_size_mb_fp32"] = model_runtime["params"] * 4 / (1024.0**2)
    if not latency_df.empty:
        out = model_runtime.merge(latency_df, on="model", how="left")
    else:
        out = model_runtime
    return out.sort_values("model")


def build_comm_efficiency_table(main_df: pd.DataFrame, default_clients: int, params: Dict[str, int]) -> pd.DataFrame:
    if main_df.empty:
        return pd.DataFrame()
    fed = main_df[main_df["setting"].isin(["FedAvg", "FedProx"])].copy()
    if fed.empty:
        return pd.DataFrame()

    fed["params"] = fed["model"].map(params)
    fed["num_clients"] = default_clients
    fed["bytes_per_round"] = fed["params"] * 4 * fed["num_clients"] * 2
    fed["total_bytes"] = fed["bytes_per_round"] * fed["global_rounds"].fillna(0)
    fed["total_mb"] = fed["total_bytes"] / (1024.0**2)
    fed["nmse_improvement_per_mb"] = np.where(
        fed["total_mb"] > 0,
        (-fed["nmse_db_mean"]) / fed["total_mb"],
        np.nan,
    )
    cols = [
        "model",
        "setting",
        "data_mode",
        "snr_db",
        "global_rounds",
        "local_epochs",
        "mu",
        "nmse_db_mean",
        "total_mb",
        "nmse_improvement_per_mb",
        "runs",
    ]
    return fed[cols].sort_values(["model", "setting", "data_mode", "global_rounds", "local_epochs", "mu"])


def build_reliability_table(run_df: pd.DataFrame, threshold_db: float) -> pd.DataFrame:
    if run_df.empty:
        return pd.DataFrame()

    rows: List[Dict[str, object]] = []
    group_cols = ["model", "setting", "data_mode", "snr_db"]
    for key, block in run_df.groupby(group_cols, dropna=False):
        vals = block["final_nmse_db"].dropna().to_numpy()
        if len(vals) == 0:
            continue
        mean_v = float(np.mean(vals))
        std_v = float(np.std(vals))
        rows.append(
            {
                "model": key[0],
                "setting": key[1],
                "data_mode": key[2],
                "snr_db": key[3],
                "runs": int(len(vals)),
                "nmse_db_mean": mean_v,
                "nmse_db_std": std_v,
                "coeff_variation_abs": abs(std_v / mean_v) if mean_v != 0 else np.nan,
                "success_rate_at_threshold": float(np.mean(vals <= threshold_db)),
                "rounds_to_threshold_mean": float(pd.to_numeric(block["rounds_to_threshold"], errors="coerce").dropna().mean())
                if block["rounds_to_threshold"].notna().any()
                else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["model", "setting", "data_mode", "snr_db"])


def _save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _record_figure(manifest_rows: List[Dict[str, str]], fig_path: Path, source_kind: str, source_reference: str) -> None:
    manifest_rows.append(
        {
            "figure_path": str(fig_path),
            "source_kind": source_kind,
            "source_reference": source_reference,
        }
    )


def plot_nmse_vs_snr(main_df: pd.DataFrame, out_path: Path) -> bool:
    if main_df.empty or main_df["snr_db"].nunique() < 2:
        return False
    plt.figure(figsize=(8, 5))
    grouped = main_df.groupby(["model", "setting", "data_mode"], dropna=False)
    for key, block in grouped:
        label = f"{key[0]} | {key[1]} | {key[2]}"
        block = block.sort_values("snr_db")
        plt.errorbar(block["snr_db"], block["nmse_db_mean"], yerr=block["ci95"], marker="o", capsize=3, label=label)
    plt.xlabel("SNR (dB)")
    plt.ylabel("NMSE (dB)")
    plt.title("NMSE vs SNR")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    return True


def plot_nmse_vs_rounds(run_df: pd.DataFrame, out_path: Path, mode: str) -> bool:
    if run_df.empty:
        return False
    fl = run_df[(run_df["setting"].isin(["FedAvg", "FedProx"])) & (run_df["data_mode"] == mode)]
    if fl.empty:
        return False

    raw_path = PROJECT_ROOT / "results" / "raw"
    curves: Dict[Tuple[str, str], Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
    for run_id in fl["run_id"].dropna().tolist():
        path = raw_path / f"{run_id}.json"
        if not path.exists():
            continue
        payload = pd.read_json(path, typ="series").to_dict()
        meta = payload.get("metadata", {})
        label = (meta.get("model", "NA"), meta.get("setting", "NA"))
        for point in payload.get("history", []):
            if point.get("step_kind") != "round":
                continue
            r = int(point.get("step_index", 0))
            nmse_db = _safe_float(point.get("nmse_db"))
            if np.isfinite(nmse_db):
                curves[label][r].append(nmse_db)
    if not curves:
        return False

    plt.figure(figsize=(8, 5))
    for label, by_round in sorted(curves.items()):
        rounds = sorted(by_round.keys())
        means = [float(np.mean(by_round[r])) for r in rounds]
        plt.plot(rounds, means, marker="o", label=f"{label[0]} | {label[1]}")
    plt.xlabel("Global rounds")
    plt.ylabel("NMSE (dB)")
    plt.title(f"NMSE vs Rounds ({mode})")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    return True


def plot_centralized_convergence(raw_runs: List[dict], out_path: Path) -> bool:
    losses: Dict[str, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
    for run in raw_runs:
        meta = run.get("metadata", {})
        if meta.get("setting") != "Centralized":
            continue
        model = str(meta.get("model", "Unknown"))
        for step in run.get("history", []):
            if step.get("step_kind") != "epoch":
                continue
            epoch = int(step.get("step_index", 0))
            loss = _safe_float(step.get("train_mse_loss"))
            if np.isfinite(loss):
                losses[model][epoch].append(loss)
    if not losses:
        return False

    plt.figure(figsize=(8, 5))
    for model, by_epoch in sorted(losses.items()):
        epochs = sorted(by_epoch.keys())
        means = [float(np.mean(by_epoch[e])) for e in epochs]
        plt.plot(epochs, means, marker="o", label=model)
    plt.xlabel("Epoch")
    plt.ylabel("Train MSE loss")
    plt.title("Centralized Convergence")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    return True


def plot_iid_vs_noniid(main_df: pd.DataFrame, out_path: Path) -> bool:
    if main_df.empty:
        return False
    fedavg = main_df[main_df["setting"] == "FedAvg"]
    if fedavg.empty:
        return False

    pairs = []
    for model, block in fedavg.groupby("model", dropna=False):
        iid = block[block["data_mode"] == "IID"]["nmse_db_mean"]
        non = block[block["data_mode"] == "Non-IID"]["nmse_db_mean"]
        if iid.empty or non.empty:
            continue
        pairs.append((model, float(iid.min()), float(non.min())))
    if not pairs:
        return False

    labels = [p[0] for p in pairs]
    iid_vals = [p[1] for p in pairs]
    non_vals = [p[2] for p in pairs]

    x = np.arange(len(labels))
    w = 0.35
    plt.figure(figsize=(8, 5))
    plt.bar(x - w / 2, iid_vals, w, label="IID")
    plt.bar(x + w / 2, non_vals, w, label="Non-IID")
    plt.xticks(x, labels)
    plt.ylabel("NMSE (dB)")
    plt.title("FedAvg: IID vs Non-IID")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    return True


def plot_mu_ablation(ablation_df: pd.DataFrame, out_path: Path) -> bool:
    if ablation_df.empty:
        return False
    fedprox = ablation_df[(ablation_df["setting"] == "FedProx") & (ablation_df["mu"].notna())]
    if fedprox.empty:
        return False
    plt.figure(figsize=(8, 5))
    for (model, mode), block in fedprox.groupby(["model", "data_mode"], dropna=False):
        block = block.sort_values("mu")
        plt.plot(block["mu"], block["nmse_db_mean"], marker="o", label=f"{model} | {mode}")
    plt.xlabel("FedProx $\\mu$")
    plt.ylabel("NMSE (dB)")
    plt.title("FedProx $\\mu$ Ablation")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    return True


def plot_local_epoch_ablation(ablation_df: pd.DataFrame, out_path: Path) -> bool:
    if ablation_df.empty:
        return False
    block = ablation_df[ablation_df["local_epochs"].notna()]
    if block.empty:
        return False
    plt.figure(figsize=(8, 5))
    for key, grp in block.groupby(["model", "setting", "data_mode"], dropna=False):
        grp = grp.sort_values("local_epochs")
        plt.plot(grp["local_epochs"], grp["nmse_db_mean"], marker="o", label=f"{key[0]}|{key[1]}|{key[2]}")
    plt.xlabel("Local epochs")
    plt.ylabel("NMSE (dB)")
    plt.title("Local Epoch Ablation")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=7)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    return True


def plot_round_ablation(ablation_df: pd.DataFrame, out_path: Path) -> bool:
    if ablation_df.empty:
        return False
    block = ablation_df[ablation_df["global_rounds"].notna()]
    if block.empty:
        return False
    plt.figure(figsize=(8, 5))
    for key, grp in block.groupby(["model", "setting", "data_mode"], dropna=False):
        grp = grp.sort_values("global_rounds")
        plt.plot(grp["global_rounds"], grp["nmse_db_mean"], marker="o", label=f"{key[0]}|{key[1]}|{key[2]}")
    plt.xlabel("Global rounds")
    plt.ylabel("NMSE (dB)")
    plt.title("Round Ablation")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=7)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    return True


def plot_comm_pareto(comm_df: pd.DataFrame, out_path: Path) -> bool:
    if comm_df.empty:
        return False
    plt.figure(figsize=(8, 5))
    for key, grp in comm_df.groupby(["model", "setting", "data_mode"], dropna=False):
        plt.scatter(grp["total_mb"], grp["nmse_db_mean"], label=f"{key[0]}|{key[1]}|{key[2]}")
    plt.xlabel("Total communication (MB)")
    plt.ylabel("NMSE (dB)")
    plt.title("Communication-Accuracy Pareto")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=7)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    return True


def plot_latency_bar(latency_df: pd.DataFrame, out_path: Path) -> bool:
    if latency_df.empty:
        return False
    plt.figure(figsize=(7, 4))
    plt.bar(latency_df["model"], latency_df["latency_ms_per_sample_mean"], yerr=latency_df["latency_ms_per_sample_std"], capsize=4)
    plt.ylabel("Latency (ms/sample)")
    plt.title(f"Inference Latency ({latency_df.iloc[0]['device']})")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    return True


def plot_nmse_cdf(run_df: pd.DataFrame, out_path: Path) -> bool:
    if run_df.empty:
        return False
    plt.figure(figsize=(8, 5))
    any_curve = False
    for key, grp in run_df.groupby(["model", "setting", "data_mode"], dropna=False):
        vals = grp["final_nmse_db"].dropna().to_numpy()
        if len(vals) < 2:
            continue
        xs = np.sort(vals)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        plt.plot(xs, ys, marker=".", linestyle="-", label=f"{key[0]}|{key[1]}|{key[2]}")
        any_curve = True
    if not any_curve:
        plt.close()
        return False
    plt.xlabel("Final NMSE (dB)")
    plt.ylabel("CDF")
    plt.title("NMSE Reliability CDF")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=7)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    return True


def write_manifest(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["figure_path", "source_kind", "source_reference"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    summary_dir = Path(args.summary_dir)
    figures_dir = Path(args.figures_dir)

    raw_runs = load_runs(args.raw_dir)
    run_df = build_run_level_df(raw_runs, threshold_db=args.reliability_threshold_db)
    main_df = build_main_comparison_table(run_df)
    ablation_df = build_ablation_table(main_df)
    significance_df = build_significance_table(run_df)

    latency_df = benchmark_latency_ms_per_sample(args.dataset_dir, args.latency_repeats, args.latency_warmup)
    params = _load_model_param_counts()
    complexity_df = build_complexity_table(run_df, latency_df)
    comm_df = build_comm_efficiency_table(main_df, args.default_clients, params)
    reliability_df = build_reliability_table(run_df, threshold_db=args.reliability_threshold_db)

    _save_csv(run_df, summary_dir / "run_level_metrics.csv")
    _save_csv(main_df, summary_dir / "main_comparison_table.csv")
    _save_csv(ablation_df, summary_dir / "ablation_table.csv")
    _save_csv(significance_df, summary_dir / "significance_tests.csv")
    _save_csv(complexity_df, summary_dir / "complexity_and_latency_table.csv")
    _save_csv(comm_df, summary_dir / "communication_efficiency_table.csv")
    _save_csv(reliability_df, summary_dir / "reliability_table.csv")

    manifest_rows: List[Dict[str, str]] = []
    plot_specs = [
        (plot_nmse_vs_snr, main_df, figures_dir / "nmse_vs_snr.png"),
        (plot_nmse_vs_rounds, run_df, figures_dir / "nmse_vs_rounds_iid.png", "IID"),
        (plot_nmse_vs_rounds, run_df, figures_dir / "nmse_vs_rounds_non_iid.png", "Non-IID"),
        (plot_centralized_convergence, raw_runs, figures_dir / "centralized_convergence_loss.png"),
        (plot_iid_vs_noniid, main_df, figures_dir / "iid_vs_non_iid.png"),
        (plot_mu_ablation, ablation_df, figures_dir / "fedprox_mu_ablation.png"),
        (plot_local_epoch_ablation, ablation_df, figures_dir / "local_epochs_ablation.png"),
        (plot_round_ablation, ablation_df, figures_dir / "rounds_ablation.png"),
        (plot_comm_pareto, comm_df, figures_dir / "communication_pareto.png"),
        (plot_latency_bar, latency_df, figures_dir / "latency_comparison.png"),
        (plot_nmse_cdf, run_df, figures_dir / "nmse_cdf.png"),
    ]

    for spec in plot_specs:
        fn = spec[0]
        if fn is plot_nmse_vs_rounds:
            ok = fn(spec[1], spec[2], spec[3])  # type: ignore[arg-type]
        else:
            ok = fn(spec[1], spec[2])  # type: ignore[arg-type]
        if ok:
            _record_figure(manifest_rows, spec[2], "raw_dir", args.raw_dir)

    write_manifest(Path(args.manifest), manifest_rows)

    print(f"Loaded runs: {len(raw_runs)}")
    print(f"Tables saved in: {summary_dir}")
    print(f"Figures saved in: {figures_dir}")
    print(f"Figure manifest: {args.manifest}")


if __name__ == "__main__":
    main()
