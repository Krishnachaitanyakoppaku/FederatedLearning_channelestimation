"""Aggregate experiment logs into paper-ready summary tables."""

from __future__ import annotations

import argparse
import math
import json
import os
from collections import defaultdict
from statistics import mean, pstdev

from research_core.logging_utils import write_summary_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate raw experiment logs")
    parser.add_argument("--raw_dir", type=str, default="results/raw", help="Directory containing run JSON logs")
    parser.add_argument("--out", type=str, default="results/summary/aggregated_metrics.csv", help="Output CSV path")
    parser.add_argument(
        "--paper_table",
        type=str,
        default="results/summary/final_comparison_table.csv",
        help="Output CSV path for final paper comparison table",
    )
    parser.add_argument(
        "--coverage_out",
        type=str,
        default="results/summary/coverage_report.csv",
        help="Output CSV path for coverage/missing-cell report",
    )
    return parser.parse_args()


def load_runs(raw_dir: str):
    runs = []
    if not os.path.isdir(raw_dir):
        return runs

    for name in os.listdir(raw_dir):
        if not name.endswith(".json"):
            continue
        path = os.path.join(raw_dir, name)
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if "metadata" in payload and "summary" in payload:
            runs.append(payload)
    return runs


def aggregate(runs):
    grouped = defaultdict(list)
    for run in runs:
        meta = run["metadata"]
        summary = run.get("summary", {})
        if "final_nmse_db" not in summary:
            continue
        key = (
            meta["model"],
            meta["setting"],
            meta["data_mode"],
            meta["snr_db"],
            meta.get("local_epochs"),
            meta.get("global_rounds"),
            meta.get("mu"),
        )
        grouped[key].append(float(summary["final_nmse_db"]))

    rows = []
    for key, values in grouped.items():
        model, setting, data_mode, snr_db, local_epochs, rounds, mu = key
        rows.append(
            {
                "model": model,
                "setting": setting,
                "data_mode": data_mode,
                "snr_db": snr_db,
                "local_epochs": local_epochs,
                "rounds": rounds,
                "mu": mu,
                "runs": len(values),
                "nmse_db_mean": round(mean(values), 6),
                "nmse_db_std": round(pstdev(values), 6) if len(values) > 1 else 0.0,
                "nmse_db_ci95": round(1.96 * (pstdev(values) / math.sqrt(len(values))), 6)
                if len(values) > 1
                else 0.0,
            }
        )
    return sorted(rows, key=lambda r: (r["model"], r["setting"], r["data_mode"], r["snr_db"]))


def build_paper_table(rows):
    centralized = {}
    for row in rows:
        if row["setting"] == "Centralized":
            centralized[(row["model"], row["snr_db"])] = row["nmse_db_mean"]

    paper_rows = []
    for row in rows:
        baseline = centralized.get((row["model"], row["snr_db"]))
        gap = None if baseline is None else round(row["nmse_db_mean"] - baseline, 6)
        paper_rows.append(
            {
                "model": row["model"],
                "setting": row["setting"],
                "data_mode": row["data_mode"],
                "snr_db": row["snr_db"],
                "nmse_db_mean": row["nmse_db_mean"],
                "nmse_db_std": row["nmse_db_std"],
                "nmse_db_ci95": row.get("nmse_db_ci95", 0.0),
                "centralized_gap_db": gap,
                "runs": row["runs"],
            }
        )
    return paper_rows


def expected_cells_from_runs(runs):
    expected = set()
    for run in runs:
        meta = run.get("metadata", {})
        key = (
            meta.get("model"),
            meta.get("setting"),
            meta.get("data_mode"),
            meta.get("snr_db"),
            meta.get("local_epochs"),
            meta.get("global_rounds"),
            meta.get("mu"),
        )
        expected.add(key)
    return expected


def coverage_report(runs, rows):
    expected = expected_cells_from_runs(runs)
    present = {
        (
            r["model"],
            r["setting"],
            r["data_mode"],
            r["snr_db"],
            r.get("local_epochs"),
            r.get("rounds"),
            r.get("mu"),
        )
        for r in rows
    }

    output = []
    for key in sorted(expected):
        model, setting, data_mode, snr_db, local_epochs, rounds, mu = key
        output.append(
            {
                "model": model,
                "setting": setting,
                "data_mode": data_mode,
                "snr_db": snr_db,
                "local_epochs": local_epochs,
                "rounds": rounds,
                "mu": mu,
                "status": "present" if key in present else "missing",
            }
        )
    return output


def main() -> None:
    args = parse_args()
    runs = load_runs(args.raw_dir)
    rows = aggregate(runs)
    write_summary_table(rows, args.out)
    write_summary_table(build_paper_table(rows), args.paper_table)
    write_summary_table(coverage_report(runs, rows), args.coverage_out)
    print(f"Aggregated {len(runs)} runs into: {args.out}")
    print(f"Paper comparison table written to: {args.paper_table}")
    print(f"Coverage report written to: {args.coverage_out}")


if __name__ == "__main__":
    main()
