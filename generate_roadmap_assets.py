"""Generate publication-grade tables and figures aligned with PUBLICATION_ROADMAP.md.

Phases:
0) Data audit
1) Canonical summaries
2) Paper tables (Table 1-4)
3) Primary figures (Figure 1-8)
4) Supplementary figures (Figure 9-12)
5) Statistical checks + claim evidence matrix
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy import stats  # type: ignore
except Exception:  # pragma: no cover
    stats = None


PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass
class Paths:
    raw_dir: Path
    summary_dir: Path
    figure_dir: Path
    paper_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build publication assets in phases")
    parser.add_argument("--raw_dir", default="results/raw", type=str)
    parser.add_argument("--summary_dir", default="results/summary", type=str)
    parser.add_argument("--figure_dir", default="results/figures", type=str)
    parser.add_argument("--paper_model", default="CNN", type=str, help="Model used for roadmap paper assets")
    parser.add_argument("--phase", default="all", choices=["all", "0", "1", "2", "3", "4", "5"], type=str)
    return parser.parse_args()


def _safe_float(value, default=np.nan) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def load_runs(raw_dir: Path) -> List[dict]:
    runs: List[dict] = []
    if not raw_dir.exists():
        return runs
    for p in sorted(raw_dir.glob("*.json")):
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if "metadata" in payload and "summary" in payload:
            runs.append(payload)
    return runs


def build_run_df(runs: Iterable[dict]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for run in runs:
        meta = run.get("metadata", {})
        summary = run.get("summary", {})
        extra = meta.get("extra") or {}
        rows.append(
            {
                "run_id": meta.get("run_id"),
                "model": meta.get("model"),
                "setting": meta.get("setting"),
                "data_mode": meta.get("data_mode"),
                "snr_db": _safe_float(meta.get("snr_db")),
                "seed": meta.get("seed"),
                "local_epochs": _safe_float(meta.get("local_epochs")),
                "global_rounds": _safe_float(meta.get("global_rounds")),
                "mu": _safe_float(meta.get("mu")),
                "lr": _safe_float(meta.get("lr")),
                "batch_size": _safe_float(meta.get("batch_size")),
                "num_clients": _safe_float(extra.get("num_clients")),
                "duration_sec": _safe_float(run.get("duration_sec")),
                "final_nmse_db": _safe_float(summary.get("final_nmse_db")),
                "final_nmse_linear": _safe_float(summary.get("final_nmse_linear")),
            }
        )
    out = pd.DataFrame(rows)
    for col in ["seed", "local_epochs", "global_rounds", "batch_size", "num_clients", "snr_db"]:
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def build_history_df(runs: Iterable[dict]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for run in runs:
        meta = run.get("metadata", {})
        run_id = meta.get("run_id")
        for step in run.get("history", []):
            rows.append(
                {
                    "run_id": run_id,
                    "setting": meta.get("setting"),
                    "data_mode": meta.get("data_mode"),
                    "seed": _safe_float(meta.get("seed")),
                    "local_epochs": _safe_float(meta.get("local_epochs")),
                    "global_rounds": _safe_float(meta.get("global_rounds")),
                    "mu": _safe_float(meta.get("mu")),
                    "step_kind": step.get("step_kind"),
                    "step_index": _safe_float(step.get("step_index")),
                    "nmse_db": _safe_float(step.get("nmse_db")),
                    "train_mse_loss": _safe_float(step.get("train_mse_loss")),
                }
            )
    return pd.DataFrame(rows)


def ensure_dirs(paths: Paths) -> None:
    paths.summary_dir.mkdir(parents=True, exist_ok=True)
    paths.figure_dir.mkdir(parents=True, exist_ok=True)
    paths.paper_dir.mkdir(parents=True, exist_ok=True)


def phase0_audit(run_df: pd.DataFrame, paths: Paths) -> None:
    expected = {
        ("CNN", "Centralized", "N/A"): 3,
        ("CNN", "FedAvg", "IID"): 48,
        ("CNN", "FedAvg", "Non-IID"): 48,
        ("CNN", "FedProx", "Non-IID"): 144,
        ("ResUNet", "Centralized", "N/A"): 3,
    }
    rows = []
    grouped = run_df.groupby(["model", "setting", "data_mode"], dropna=False).size().to_dict()
    for key, exp in expected.items():
        got = int(grouped.get(key, 0))
        rows.append(
            {
                "model": key[0],
                "setting": key[1],
                "data_mode": key[2],
                "expected_runs": exp,
                "observed_runs": got,
                "status": "ok" if got == exp else "mismatch",
            }
        )
    audit_df = pd.DataFrame(rows)
    audit_df.to_csv(paths.summary_dir / "data_audit.csv", index=False)

    lines = [
        "# Data Audit",
        "",
        f"- Total runs: {len(run_df)}",
        f"- Missing final NMSE: {int(run_df['final_nmse_db'].isna().sum())}",
        "",
        "| Model | Setting | Mode | Expected | Observed | Status |",
        "|---|---|---|---:|---:|---|",
    ]
    for _, r in audit_df.iterrows():
        lines.append(
            f"| {r['model']} | {r['setting']} | {r['data_mode']} | {int(r['expected_runs'])} | {int(r['observed_runs'])} | {r['status']} |"
        )
    (paths.summary_dir / "data_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def phase1_canonical_summary(run_df: pd.DataFrame, paths: Paths) -> None:
    run_df.to_csv(paths.summary_dir / "run_level_metrics.csv", index=False)

    group_cols = ["model", "setting", "data_mode", "snr_db", "local_epochs", "global_rounds", "mu"]
    main_df = (
        run_df.groupby(group_cols, dropna=False)["final_nmse_db"]
        .agg(["count", "mean", "std", "min", "max"])
        .reset_index()
        .rename(columns={"count": "runs", "mean": "nmse_db_mean", "std": "nmse_db_std", "min": "nmse_db_min", "max": "nmse_db_max"})
    )
    main_df["nmse_db_std"] = main_df["nmse_db_std"].fillna(0.0)
    main_df.to_csv(paths.summary_dir / "main_comparison_table.csv", index=False)

    final_cols = ["model", "setting", "data_mode", "snr_db", "nmse_db_mean", "nmse_db_std", "runs"]
    final_df = (
        main_df[final_cols]
        .groupby(["model", "setting", "data_mode", "snr_db"], dropna=False)
        .agg({"nmse_db_mean": "mean", "nmse_db_std": "mean", "runs": "sum"})
        .reset_index()
    )
    final_df.to_csv(paths.summary_dir / "final_comparison_table.csv", index=False)

    coverage_df = run_df[["model", "setting", "data_mode", "snr_db", "local_epochs", "global_rounds", "mu"]].copy()
    coverage_df["status"] = "present"
    coverage_df = coverage_df.drop_duplicates().sort_values(["model", "setting", "data_mode"])
    coverage_df.to_csv(paths.summary_dir / "coverage_report.csv", index=False)


def _group_slice(df: pd.DataFrame, setting: str, mode: str) -> pd.Series:
    return df[(df["setting"] == setting) & (df["data_mode"] == mode)]["final_nmse_db"].dropna()


def phase2_tables(run_df: pd.DataFrame, paths: Paths) -> None:
    groups = [
        ("Centralized", "N/A"),
        ("FedAvg", "IID"),
        ("FedAvg", "Non-IID"),
        ("FedProx", "Non-IID"),
    ]

    # Table 1: best config per group
    rows_1 = []
    for setting, mode in groups:
        block = run_df[(run_df["setting"] == setting) & (run_df["data_mode"] == mode)].copy()
        if block.empty:
            continue
        best = block.loc[block["final_nmse_db"].idxmin()]
        cfg_bits = []
        if pd.notna(best["global_rounds"]):
            cfg_bits.append(f"R{int(best['global_rounds'])}")
        if pd.notna(best["local_epochs"]):
            cfg_bits.append(f"E{int(best['local_epochs'])}")
        if pd.notna(best["mu"]):
            cfg_bits.append(f"mu={best['mu']:.3g}")
        cfg_bits.append(f"seed{int(best['seed'])}")
        rows_1.append(
            {
                "algorithm": setting,
                "mode": mode,
                "nmse_db": round(float(best["final_nmse_db"]), 4),
                "config": ", ".join(cfg_bits),
                "run_id": best["run_id"],
            }
        )
    pd.DataFrame(rows_1).to_csv(paths.summary_dir / "paper_table_1_best_configs.csv", index=False)

    # Table 2: statistical summary
    central_mean = float(_group_slice(run_df, "Centralized", "N/A").mean())
    rows_2 = []
    for setting, mode in groups:
        vals = _group_slice(run_df, setting, mode)
        if vals.empty:
            continue
        rows_2.append(
            {
                "algorithm": f"{setting}-{mode}" if mode != "N/A" else setting,
                "N": int(len(vals)),
                "mean_nmse_db": round(float(vals.mean()), 4),
                "std_nmse_db": round(float(vals.std(ddof=0)), 4),
                "min_nmse_db": round(float(vals.min()), 4),
                "max_nmse_db": round(float(vals.max()), 4),
                "vs_centralized_db": round(float(vals.mean() - central_mean), 4),
            }
        )
    pd.DataFrame(rows_2).to_csv(paths.summary_dir / "paper_table_2_statistical_summary.csv", index=False)

    # Table 3: hyperparameter ablation
    fed = run_df[(run_df["setting"].isin(["FedAvg", "FedProx"])) & (run_df["data_mode"] != "N/A")]
    rows_3 = []
    for var, col in [
        ("Global Rounds", "global_rounds"),
        ("Local Epochs", "local_epochs"),
        ("FedProx mu", "mu"),
        ("Random Seed", "seed"),
    ]:
        src = fed if col != "mu" else fed[fed["setting"] == "FedProx"]
        g = src.groupby(col, dropna=True)["final_nmse_db"].mean().dropna()
        if g.empty:
            continue
        best_key = g.idxmin()
        rows_3.append(
            {
                "variable": var,
                "range": ",".join([str(x) for x in sorted(g.index.tolist())]),
                "performance_range_db": f"{g.min():.4f} to {g.max():.4f}",
                "best": str(best_key),
                "sensitivity_span_db": round(float(g.max() - g.min()), 4),
            }
        )
    pd.DataFrame(rows_3).to_csv(paths.summary_dir / "paper_table_3_hyperparameter_ablation.csv", index=False)

    # Table 4: communication cost analysis (fixed showcase rows)
    rows_4 = []
    for r, e in [(10, 1), (30, 3), (50, 5)]:
        sub = run_df[
            (run_df["setting"] == "FedAvg")
            & (run_df["data_mode"] == "IID")
            & (run_df["global_rounds"] == r)
            & (run_df["local_epochs"] == e)
        ]
        if sub.empty:
            continue
        nmse = float(sub["final_nmse_db"].mean())
        rows_4.append(
            {
                "config": f"FedAvg-IID (R{r},E{e})",
                "R": r,
                "E": e,
                "comm_events": int(r * 5),
                "nmse_db": round(nmse, 4),
                "efficiency": "High" if r >= 50 else ("Medium" if r >= 30 else "Low"),
            }
        )
    central = _group_slice(run_df, "Centralized", "N/A")
    if not central.empty:
        rows_4.append(
            {
                "config": "Centralized (20ep)",
                "R": np.nan,
                "E": 20,
                "comm_events": 0,
                "nmse_db": round(float(central.mean()), 4),
                "efficiency": "N/A",
            }
        )
    pd.DataFrame(rows_4).to_csv(paths.summary_dir / "paper_table_4_communication_cost.csv", index=False)


def _save_fig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    if path.suffix.lower() == ".png":
        pdf_path = path.with_suffix(".pdf")
        plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close()


def phase3_primary_figures(run_df: pd.DataFrame, hist_df: pd.DataFrame, paths: Paths) -> None:
    cmap = "tab10"

    # Figure 1
    cent = run_df[(run_df["setting"] == "Centralized") & (run_df["model"] == "CNN")].sort_values("seed")
    plt.figure(figsize=(5, 4))
    plt.bar(cent["seed"].astype(int).astype(str), cent["final_nmse_db"], color="#1f77b4")
    m, s = float(cent["final_nmse_db"].mean()), float(cent["final_nmse_db"].std(ddof=0))
    plt.errorbar([1], [m], yerr=[s], fmt="none", ecolor="black", capsize=5)
    plt.ylabel("NMSE (dB)")
    plt.xlabel("Seed")
    plt.ylim(-19.8, -19.0)
    plt.title("Figure 1: Centralized Baseline")
    _save_fig(paths.paper_dir / "figure1_centralized_baseline.png")

    # Figure 2
    fed_iid = run_df[(run_df["setting"] == "FedAvg") & (run_df["data_mode"] == "IID")]
    plt.figure(figsize=(7, 4.5))
    for e in [1, 2, 3, 5]:
        block = fed_iid[fed_iid["local_epochs"] == e]
        g = block.groupby("global_rounds")["final_nmse_db"].agg(["mean", "std", "count"]).sort_index()
        ci = 1.96 * (g["std"].fillna(0.0) / np.sqrt(g["count"]))
        plt.plot(g.index, g["mean"], marker="o", label=f"E{e}")
        plt.fill_between(g.index, g["mean"] - ci, g["mean"] + ci, alpha=0.15)
    plt.xlabel("Rounds")
    plt.ylabel("NMSE (dB)")
    plt.title("Figure 2: Global Rounds Impact (FedAvg-IID)")
    plt.grid(alpha=0.3)
    plt.legend()
    _save_fig(paths.paper_dir / "figure2_rounds_impact_fedavg_iid.png")

    # Figure 3
    pivot = fed_iid.groupby(["local_epochs", "global_rounds"])["final_nmse_db"].mean().unstack()
    plt.figure(figsize=(6, 4))
    x = np.arange(len(pivot.index))
    w = 0.18
    for i, r in enumerate([10, 20, 30, 50]):
        if r in pivot.columns:
            plt.bar(x + (i - 1.5) * w, pivot[r].values, width=w, label=f"R{r}")
    overall = fed_iid.groupby("local_epochs")["final_nmse_db"].mean().reindex([1, 2, 3, 5])
    plt.plot(x, overall.values, color="black", marker="o", linewidth=2, label="Mean across rounds")
    plt.xticks(x, ["1", "2", "3", "5"])
    plt.xlabel("Local Epochs")
    plt.ylabel("Mean NMSE (dB)")
    plt.title("Figure 3: Local Epochs Impact")
    plt.legend(fontsize=8)
    _save_fig(paths.paper_dir / "figure3_local_epochs_impact.png")

    # Figure 4
    iid_vals = _group_slice(run_df, "FedAvg", "IID")
    non_vals = _group_slice(run_df, "FedAvg", "Non-IID")
    c_best = float(_group_slice(run_df, "Centralized", "N/A").min())
    plt.figure(figsize=(7, 5))
    plt.boxplot([iid_vals, non_vals], tick_labels=["FedAvg-IID", "FedAvg-NonIID"], patch_artist=True)
    plt.axhline(c_best, color="red", linestyle="--", label=f"Centralized best ({c_best:.2f} dB)")
    plt.ylabel("Final NMSE (dB)")
    plt.title("Figure 4: IID vs NonIID Comparison")
    plt.legend()
    _save_fig(paths.paper_dir / "figure4_iid_vs_noniid.png")

    # Figure 5 (variance reduction heatmap)
    fa_var = (
        run_df[(run_df["setting"] == "FedAvg") & (run_df["data_mode"] == "Non-IID")]
        .groupby(["global_rounds", "local_epochs"]) ["final_nmse_db"]
        .std(ddof=0)
    )
    fig, axes = plt.subplots(1, 3, figsize=(10, 4), sharey=True)
    for idx, mu in enumerate([0.001, 0.005, 0.01]):
        fp_var = (
            run_df[(run_df["setting"] == "FedProx") & (run_df["data_mode"] == "Non-IID") & (np.isclose(run_df["mu"], mu))]
            .groupby(["global_rounds", "local_epochs"]) ["final_nmse_db"]
            .std(ddof=0)
        )
        rows, cols = [10, 20, 30, 50], [1, 2, 3, 5]
        mat = np.full((len(rows), len(cols)), np.nan)
        for i, r in enumerate(rows):
            for j, e in enumerate(cols):
                a = fa_var.get((r, e), np.nan)
                b = fp_var.get((r, e), np.nan)
                if np.isfinite(a) and a > 0 and np.isfinite(b):
                    mat[i, j] = 100.0 * (a - b) / a
        im = axes[idx].imshow(mat, cmap="coolwarm", aspect="auto", vmin=-100, vmax=100)
        axes[idx].set_title(f"mu={mu}")
        axes[idx].set_xticks(range(len(cols)), cols)
        axes[idx].set_yticks(range(len(rows)), rows)
        axes[idx].set_xlabel("Local epochs")
        if idx == 0:
            axes[idx].set_ylabel("Rounds")
    fig.colorbar(im, ax=axes.ravel().tolist(), label="Variance reduction (%)")
    fig.suptitle("Figure 5: FedProx Variance Reduction vs FedAvg-NonIID")
    _save_fig(paths.paper_dir / "figure5_fedprox_variance_reduction.png")

    # Figure 6
    fig, axes = plt.subplots(1, 3, figsize=(11, 4), sharey=True)
    rows, cols = [10, 20, 30, 50], [1, 2, 3, 5]
    for idx, mu in enumerate([0.001, 0.005, 0.01]):
        block = run_df[(run_df["setting"] == "FedProx") & (run_df["data_mode"] == "Non-IID") & (np.isclose(run_df["mu"], mu))]
        pvt = block.pivot_table(values="final_nmse_db", index="global_rounds", columns="local_epochs", aggfunc="mean")
        mat = np.full((len(rows), len(cols)), np.nan)
        for i, r in enumerate(rows):
            for j, e in enumerate(cols):
                if r in pvt.index and e in pvt.columns:
                    mat[i, j] = pvt.loc[r, e]
        im = axes[idx].imshow(mat, cmap="RdYlBu", aspect="auto", vmin=-20.2, vmax=-17.0)
        axes[idx].set_title(f"mu={mu}")
        axes[idx].set_xticks(range(len(cols)), cols)
        axes[idx].set_yticks(range(len(rows)), rows)
        axes[idx].set_xlabel("Local epochs")
        if idx == 0:
            axes[idx].set_ylabel("Rounds")
    fig.colorbar(im, ax=axes.ravel().tolist(), label="NMSE (dB)")
    fig.suptitle("Figure 6: FedProx Hyperparameter Heatmap")
    _save_fig(paths.paper_dir / "figure6_fedprox_heatmap.png")

    # Figure 7
    labels = ["Centralized", "FedAvg-IID", "FedAvg-NonIID", "FedProx-NonIID"]
    slices = [
        _group_slice(run_df, "Centralized", "N/A"),
        _group_slice(run_df, "FedAvg", "IID"),
        _group_slice(run_df, "FedAvg", "Non-IID"),
        _group_slice(run_df, "FedProx", "Non-IID"),
    ]
    best = [float(s.min()) for s in slices]
    mean = [float(s.mean()) for s in slices]
    worst = [float(s.max()) for s in slices]
    x = np.arange(len(labels))
    w = 0.24
    plt.figure(figsize=(8, 5))
    plt.bar(x - w, best, width=w, label="Best", color="#2ca02c")
    plt.bar(x, mean, width=w, label="Mean", color="#1f77b4")
    plt.bar(x + w, worst, width=w, label="Worst", color="#ff7f0e")
    plt.axhline(best[0], color="red", linestyle="--", linewidth=1)
    plt.xticks(x, labels, rotation=10)
    plt.ylabel("NMSE (dB)")
    plt.title("Figure 7: Algorithm Comparison (Best/Mean/Worst)")
    plt.legend()
    _save_fig(paths.paper_dir / "figure7_algorithm_summary.png")

    # Figure 8
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    round_hist = hist_df[hist_df["step_kind"] == "round"].copy()

    # A: IID lines from R=50 runs, E in {1,3,5}
    for e in [1, 3, 5]:
        block = round_hist[
            (round_hist["setting"] == "FedAvg")
            & (round_hist["data_mode"] == "IID")
            & (round_hist["global_rounds"] == 50)
            & (round_hist["local_epochs"] == e)
        ]
        g = block.groupby("step_index")["nmse_db"].mean().sort_index()
        axes[0].plot(g.index, g.values, label=f"FedAvg IID E{e}")
    axes[0].axhline(float(_group_slice(run_df, "Centralized", "N/A").mean()), linestyle="--", color="black", label="Centralized mean")
    axes[0].set_title("Figure 8A: Convergence (IID)")
    axes[0].set_xlabel("Global rounds")
    axes[0].set_ylabel("Average NMSE (dB)")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)

    # B: Non-IID, R=50, E=5
    fa_block = round_hist[
        (round_hist["setting"] == "FedAvg")
        & (round_hist["data_mode"] == "Non-IID")
        & (round_hist["global_rounds"] == 50)
        & (round_hist["local_epochs"] == 5)
    ]
    g = fa_block.groupby("step_index")["nmse_db"].mean().sort_index()
    axes[1].plot(g.index, g.values, label="FedAvg Non-IID")
    for mu in [0.001, 0.005, 0.01]:
        fp_block = round_hist[
            (round_hist["setting"] == "FedProx")
            & (round_hist["data_mode"] == "Non-IID")
            & (round_hist["global_rounds"] == 50)
            & (round_hist["local_epochs"] == 5)
            & (np.isclose(round_hist["mu"], mu))
        ]
        gm = fp_block.groupby("step_index")["nmse_db"].mean().sort_index()
        axes[1].plot(gm.index, gm.values, label=f"FedProx mu={mu}")
    axes[1].set_title("Figure 8B: Convergence (Non-IID)")
    axes[1].set_xlabel("Global rounds")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    _save_fig(paths.paper_dir / "figure8_convergence_curves.png")


def phase4_supplementary_figures(run_df: pd.DataFrame, paths: Paths) -> None:
    # Figure 9: seed robustness for 16 configs at mu=0.001
    fp = run_df[(run_df["setting"] == "FedProx") & (run_df["data_mode"] == "Non-IID") & (np.isclose(run_df["mu"], 0.001))]
    pvt = fp.pivot_table(values="final_nmse_db", index=["global_rounds", "local_epochs"], columns="seed")
    pvt = pvt.sort_index()
    plt.figure(figsize=(6, 5))
    plt.imshow(pvt.values, cmap="RdYlBu", aspect="auto")
    plt.colorbar(label="NMSE (dB)")
    plt.yticks(range(len(pvt.index)), [f"R{int(r)} E{int(e)}" for r, e in pvt.index], fontsize=7)
    plt.xticks(range(len(pvt.columns)), [int(c) for c in pvt.columns])
    plt.title("Figure 9: Seed Robustness (FedProx mu=0.001)")
    _save_fig(paths.paper_dir / "figure9_seed_robustness.png")

    # Figure 10: communication efficiency
    fed = run_df[run_df["setting"].isin(["FedAvg", "FedProx"])].copy()
    fed["comm_events"] = fed["global_rounds"] * fed["num_clients"].fillna(5)
    fed["alg"] = fed["setting"] + "-" + fed["data_mode"]
    plt.figure(figsize=(6.5, 4.5))
    for alg, block in fed.groupby("alg"):
        plt.scatter(block["comm_events"], block["final_nmse_db"], s=20 + 15 * block["local_epochs"], alpha=0.7, label=alg)
    plt.xlabel("Total communication events (rounds x clients)")
    plt.ylabel("Final NMSE (dB)")
    plt.title("Figure 10: Communication Efficiency")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=7)
    _save_fig(paths.paper_dir / "figure10_communication_efficiency.png")

    # Figure 11: contour for mu=0.001
    mean_grid = fp.groupby(["global_rounds", "local_epochs"])["final_nmse_db"].mean().unstack()
    x = mean_grid.columns.values
    y = mean_grid.index.values
    xx, yy = np.meshgrid(x, y)
    zz = mean_grid.values
    plt.figure(figsize=(7, 6))
    c = plt.contourf(xx, yy, zz, levels=12, cmap="viridis")
    plt.colorbar(c, label="NMSE (dB)")
    plt.xlabel("Local epochs")
    plt.ylabel("Global rounds")
    plt.title("Figure 11: Epoch-Rounds Surface (FedProx mu=0.001)")
    _save_fig(paths.paper_dir / "figure11_epoch_rounds_surface.png")

    # Figure 12: variance decomposition
    groups = [
        ("Centralized", "N/A", "Centralized"),
        ("FedAvg", "IID", "FedAvg-IID"),
        ("FedAvg", "Non-IID", "FedAvg-NonIID"),
        ("FedProx", "Non-IID", "FedProx-NonIID"),
    ]
    labels, seed_var, cfg_var = [], [], []
    for setting, mode, label in groups:
        sub = run_df[(run_df["setting"] == setting) & (run_df["data_mode"] == mode)]
        labels.append(label)
        by_cfg = sub.groupby(["global_rounds", "local_epochs", "mu"], dropna=False)["final_nmse_db"]
        within = by_cfg.var(ddof=0).dropna()
        seed_var.append(float(within.mean()) if len(within) else 0.0)
        cfg_mean = by_cfg.mean().dropna()
        cfg_var.append(float(cfg_mean.var(ddof=0)) if len(cfg_mean) else 0.0)
    x = np.arange(len(labels))
    plt.figure(figsize=(6, 4))
    plt.bar(x, seed_var, label="Across-seed variance")
    plt.bar(x, cfg_var, bottom=seed_var, label="Across-config variance")
    plt.xticks(x, labels, rotation=10)
    plt.ylabel("Variance (dB^2)")
    plt.title("Figure 12: Variance Decomposition")
    plt.legend(fontsize=8)
    _save_fig(paths.paper_dir / "figure12_variance_decomposition.png")


def _welch(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    if stats is None:
        return np.nan, np.nan
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return float(t), float(p)


def phase5_stats_and_claims(run_df: pd.DataFrame, paths: Paths) -> None:
    c = _group_slice(run_df, "Centralized", "N/A").to_numpy()
    fi = _group_slice(run_df, "FedAvg", "IID").to_numpy()
    fn = _group_slice(run_df, "FedAvg", "Non-IID").to_numpy()
    fp = _group_slice(run_df, "FedProx", "Non-IID").to_numpy()

    rows = []
    for name_a, a, name_b, b in [
        ("FedAvg-IID", fi, "Centralized", c),
        ("FedAvg-IID", fi, "FedAvg-NonIID", fn),
        ("FedAvg-NonIID", fn, "FedProx-NonIID", fp),
    ]:
        t, p = _welch(a, b)
        rows.append(
            {
                "group_a": name_a,
                "group_b": name_b,
                "mean_a": round(float(np.mean(a)), 4),
                "mean_b": round(float(np.mean(b)), 4),
                "delta_a_minus_b_db": round(float(np.mean(a) - np.mean(b)), 4),
                "welch_t": round(t, 4) if np.isfinite(t) else np.nan,
                "welch_pvalue": round(p, 6) if np.isfinite(p) else np.nan,
            }
        )
    stat_df = pd.DataFrame(rows)
    stat_df.to_csv(paths.summary_dir / "statistical_tests_research_grade.csv", index=False)

    claim_lines = [
        "# Claim-Evidence Matrix",
        "",
        "| Claim | Evidence | Metric |",
        "|---|---|---|",
        "| FedAvg-IID reaches/beats centralized baseline | Figure 1, Figure 2, Figure 7, Table 1/2 | Best and mean NMSE deltas |",
        "| Non-IID introduces measurable penalty | Figure 4, Figure 7, Table 2 | Distribution shift and mean gap |",
        "| FedProx improves robustness under Non-IID | Figure 5, Figure 8B, Table 2/3 | Variance reduction and smoother convergence |",
        "| Communication-performance trade-off is explicit | Figure 10, Table 4 | NMSE vs communication events |",
    ]
    (paths.summary_dir / "claim_evidence_matrix.md").write_text("\n".join(claim_lines) + "\n", encoding="utf-8")

    report_lines = [
        "# Statistical Check Report",
        "",
        "- Tests: Welch t-test (two-sided)",
        "- Source: `results/summary/statistical_tests_research_grade.csv`",
        "",
    ]
    for _, r in stat_df.iterrows():
        report_lines.append(
            f"- {r['group_a']} vs {r['group_b']}: delta={r['delta_a_minus_b_db']:.4f} dB, p={r['welch_pvalue']}"
        )
    (paths.summary_dir / "stat_check_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def write_phase_manifest(paths: Paths) -> None:
    lines = [
        "# Publication Build Manifest",
        "",
        "## Generated summary files",
    ]
    for p in sorted(paths.summary_dir.glob("*")):
        lines.append(f"- `{p.relative_to(PROJECT_ROOT)}`")
    lines.append("")
    lines.append("## Generated figure files")
    for p in sorted(paths.paper_dir.glob("*")):
        lines.append(f"- `{p.relative_to(PROJECT_ROOT)}`")
    (paths.summary_dir / "publication_build_manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    paths = Paths(
        raw_dir=(PROJECT_ROOT / args.raw_dir),
        summary_dir=(PROJECT_ROOT / args.summary_dir),
        figure_dir=(PROJECT_ROOT / args.figure_dir),
        paper_dir=(PROJECT_ROOT / args.figure_dir / "paper"),
    )
    ensure_dirs(paths)

    runs = load_runs(paths.raw_dir)
    run_df = build_run_df(runs)
    hist_df = build_history_df(runs)

    paper_df = run_df[run_df["model"] == args.paper_model].copy()
    paper_hist_df = hist_df[hist_df["run_id"].isin(set(paper_df["run_id"].dropna().tolist()))].copy()

    phases = [args.phase] if args.phase != "all" else ["0", "1", "2", "3", "4", "5"]
    for ph in phases:
        if ph == "0":
            phase0_audit(run_df, paths)
        elif ph == "1":
            phase1_canonical_summary(run_df, paths)
        elif ph == "2":
            phase2_tables(paper_df, paths)
        elif ph == "3":
            phase3_primary_figures(paper_df, paper_hist_df, paths)
        elif ph == "4":
            phase4_supplementary_figures(paper_df, paths)
        elif ph == "5":
            phase5_stats_and_claims(paper_df, paths)

    write_phase_manifest(paths)
    print(f"Runs loaded: {len(run_df)}")
    print(f"Summary dir: {paths.summary_dir}")
    print(f"Figure dir: {paths.paper_dir}")
    print(f"Phases completed: {', '.join(phases)}")


if __name__ == "__main__":
    main()
