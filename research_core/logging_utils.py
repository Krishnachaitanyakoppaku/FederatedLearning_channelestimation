"""Structured logging utilities for experiment reproducibility."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional


@dataclass
class RunMetadata:
    run_id: str
    model: str
    setting: str
    data_mode: str
    snr_db: int
    seed: int
    local_epochs: Optional[int]
    global_rounds: Optional[int]
    mu: Optional[float]
    lr: float
    batch_size: int
    config_path: Optional[str] = None
    config_sha256: Optional[str] = None
    git_commit: Optional[str] = None
    python_version: Optional[str] = None
    torch_version: Optional[str] = None
    device: Optional[str] = None
    platform: Optional[str] = None
    extra: Optional[Dict[str, object]] = None


def file_sha256(path: str) -> Optional[str]:
    if not path or not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def get_git_commit(start_dir: Optional[str] = None) -> Optional[str]:
    try:
        args = ["git", "rev-parse", "HEAD"]
        result = subprocess.run(
            args,
            cwd=start_dir,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def detect_torch_version() -> Optional[str]:
    try:
        import torch  # type: ignore

        return torch.__version__
    except Exception:
        return None


def runtime_platform() -> str:
    return platform.platform()


def runtime_python_version() -> str:
    return sys.version.split()[0]


class ExperimentLogger:
    """Writes machine-readable logs for per-step and summary metrics."""

    def __init__(self, root_dir: str = "results/raw"):
        self.root_dir = root_dir
        os.makedirs(self.root_dir, exist_ok=True)

    def _json_path(self, run_id: str) -> str:
        return os.path.join(self.root_dir, f"{run_id}.json")

    def _csv_path(self, run_id: str) -> str:
        return os.path.join(self.root_dir, f"{run_id}_history.csv")

    def start_run(self, metadata: RunMetadata) -> None:
        payload = {
            "metadata": metadata.__dict__,
            "started_at": time.time(),
            "history": [],
            "summary": {},
        }
        with open(self._json_path(metadata.run_id), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def log_step(
        self,
        run_id: str,
        step_kind: str,
        step_index: int,
        metrics: Dict[str, float],
    ) -> None:
        json_path = self._json_path(run_id)
        with open(json_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        row = {"step_kind": step_kind, "step_index": step_index, **metrics}
        payload["history"].append(row)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        csv_path = self._csv_path(run_id)
        file_exists = os.path.exists(csv_path)
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    def end_run(self, run_id: str, summary: Dict[str, float]) -> None:
        json_path = self._json_path(run_id)
        with open(json_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        payload["summary"] = summary
        payload["ended_at"] = time.time()
        payload["duration_sec"] = payload["ended_at"] - payload["started_at"]

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


def write_summary_table(rows: Iterable[Dict[str, object]], out_path: str) -> None:
    rows = list(rows)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if not rows:
        return

    keys: List[str] = list(rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
