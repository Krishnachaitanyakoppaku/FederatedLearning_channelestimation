"""Experiment configuration loader for paper-grade reproducibility."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class ExperimentConfig:
    seeds: List[int]
    snr_db: int
    centralized_epochs: Dict[str, int]
    batch_size: int
    lr: Dict[str, float]
    fl_rounds: List[int]
    local_epochs: List[int]
    fedprox_mu: List[float]


def load_config(path: str) -> ExperimentConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return ExperimentConfig(
        seeds=data["seeds"],
        snr_db=data["snr_db"],
        centralized_epochs=data["centralized_epochs"],
        batch_size=data["batch_size"],
        lr=data["lr"],
        fl_rounds=data["fl_rounds"],
        local_epochs=data["local_epochs"],
        fedprox_mu=data["fedprox_mu"],
    )
