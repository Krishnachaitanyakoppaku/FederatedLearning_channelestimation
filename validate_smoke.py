"""Lightweight smoke checks for pipeline readiness."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def check_dataset(dataset_root: Path) -> None:
    snrs = [0, 5, 10, 15, 20]
    for snr in snrs:
        d = dataset_root / f"{snr}dB"
        assert d.exists(), f"Missing dataset folder: {d}"
        x_train = np.load(d / "X_train.npy")
        y_train = np.load(d / "Y_train.npy")
        x_test = np.load(d / "X_test.npy")
        y_test = np.load(d / "Y_test.npy")
        stats = np.load(d / "norm_stats.npz")

        assert x_train.shape == y_train.shape == (4800, 2, 64, 64)
        assert x_test.shape == y_test.shape == (1200, 2, 64, 64)
        assert x_train.dtype == np.float32
        assert y_train.dtype == np.float32
        assert float(stats["std"]) > 0.0


def check_raw_logs(raw_dir: Path) -> None:
    if not raw_dir.exists():
        return
    for p in raw_dir.glob("*.json"):
        with p.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        assert "metadata" in payload, f"Missing metadata in {p}"
        assert "summary" in payload, f"Missing summary in {p}"


def check_results_dirs(project_root: Path) -> None:
    summary_dir = project_root / "results" / "summary"
    figures_dir = project_root / "results" / "figures"
    checkpoints_dir = project_root / "results" / "checkpoints"
    os.makedirs(summary_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run smoke checks for project readiness")
    parser.add_argument("--project_root", type=str, default=".", help="Project root path")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    check_dataset(project_root / "dataset" / "cnn")
    check_raw_logs(project_root / "results" / "raw")
    check_results_dirs(project_root)
    print("Smoke checks passed")


if __name__ == "__main__":
    main()
