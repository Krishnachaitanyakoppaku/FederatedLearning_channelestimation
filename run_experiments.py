"""Phase-wise orchestration for centralized and federated experiment runs."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from research_core.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parent
PYTHON_BIN = sys.executable


def run_cmd(command, workdir: Path):
    print(f"\n[RUN] {' '.join(command)}")
    subprocess.run(command, cwd=str(workdir), check=True)


def run_phase1(config_path: Path):
    cfg = load_config(str(config_path))
    print("Phase 1 ready: protocol/config loaded")
    print(f"Seeds={cfg.seeds}, SNR={cfg.snr_db}, rounds={cfg.fl_rounds}, local_epochs={cfg.local_epochs}")


def run_phase2(config_path: Path):
    cfg = load_config(str(config_path))
    for seed in cfg.seeds:
        run_cmd(
            [
                PYTHON_BIN,
                "train_and_evaluate.py",
                "--snr",
                str(cfg.snr_db),
                "--epochs",
                str(cfg.centralized_epochs["cnn"]),
                "--batch",
                str(cfg.batch_size),
                "--lr",
                str(cfg.lr["cnn"]),
                "--seed",
                str(seed),
                "--config_path",
                str(config_path),
            ],
            PROJECT_ROOT / "Centralised Learning" / "CNN",
        )

        run_cmd(
            [
                PYTHON_BIN,
                "resunet_model.py",
                "--snr",
                str(cfg.snr_db),
                "--epochs",
                str(cfg.centralized_epochs["resunet"]),
                "--batch",
                str(cfg.batch_size),
                "--lr",
                str(cfg.lr["resunet"]),
                "--seed",
                str(seed),
                "--config_path",
                str(config_path),
            ],
            PROJECT_ROOT / "Centralised Learning" / "ResUNet",
        )


def run_phase3(config_path: Path):
    cfg = load_config(str(config_path))

    x_train = f"../../dataset/cnn/{cfg.snr_db}dB/X_train.npy"
    y_train = f"../../dataset/cnn/{cfg.snr_db}dB/Y_train.npy"
    x_test = f"../../dataset/cnn/{cfg.snr_db}dB/X_test.npy"
    y_test = f"../../dataset/cnn/{cfg.snr_db}dB/Y_test.npy"

    run_cmd(
        [
            PYTHON_BIN,
            "split_clients.py",
            "--x_path",
            x_train,
            "--y_path",
            y_train,
            "--clients",
            "5",
            "--output_dir",
            "clients",
        ],
        PROJECT_ROOT / "Federated Learning" / "CNN_Fedavg",
    )
    run_cmd(
        [
            PYTHON_BIN,
            "non_iid_split.py",
            "--snr",
            str(cfg.snr_db),
            "--clients",
            "5",
            "--samples_per_client",
            "960",
            "--output_dir",
            "clients_non_iid",
        ],
        PROJECT_ROOT / "Federated Learning" / "CNN_FedProx",
    )

    for seed in cfg.seeds:
        for rounds in cfg.fl_rounds:
            for local_epochs in cfg.local_epochs:
                run_cmd(
                    [
                        PYTHON_BIN,
                        "federated_train.py",
                        "--rounds",
                        str(rounds),
                        "--local_epochs",
                        str(local_epochs),
                        "--batch",
                        str(cfg.batch_size),
                        "--lr",
                        str(cfg.lr["federated"]),
                        "--seed",
                        str(seed),
                        "--snr",
                        str(cfg.snr_db),
                        "--x_test",
                        x_test,
                        "--y_test",
                        y_test,
                        "--data_mode",
                        "IID",
                        "--config_path",
                        str(config_path),
                        "--checkpoint_dir",
                        str(PROJECT_ROOT / "results" / "checkpoints"),
                    ],
                    PROJECT_ROOT / "Federated Learning" / "CNN_Fedavg",
                )

                run_cmd(
                    [
                        PYTHON_BIN,
                        "federated_train.py",
                        "--rounds",
                        str(rounds),
                        "--local_epochs",
                        str(local_epochs),
                        "--batch",
                        str(cfg.batch_size),
                        "--lr",
                        str(cfg.lr["federated"]),
                        "--seed",
                        str(seed),
                        "--snr",
                        str(cfg.snr_db),
                        "--x_test",
                        x_test,
                        "--y_test",
                        y_test,
                        "--data_mode",
                        "Non-IID",
                        "--client_dir",
                        "clients_non_iid",
                        "--save",
                        f"fedavg_non_iid_r{rounds}_e{local_epochs}_s{seed}.pth",
                        "--config_path",
                        str(config_path),
                        "--checkpoint_dir",
                        str(PROJECT_ROOT / "results" / "checkpoints"),
                    ],
                    PROJECT_ROOT / "Federated Learning" / "CNN_FedProx",
                )

                for mu in cfg.fedprox_mu:
                    run_cmd(
                        [
                            PYTHON_BIN,
                            "federated_train.py",
                            "--fedprox",
                            "--mu",
                            str(mu),
                            "--rounds",
                            str(rounds),
                            "--local_epochs",
                            str(local_epochs),
                            "--batch",
                            str(cfg.batch_size),
                            "--lr",
                            str(cfg.lr["federated"]),
                            "--seed",
                            str(seed),
                            "--snr",
                            str(cfg.snr_db),
                            "--x_test",
                            x_test,
                            "--y_test",
                            y_test,
                            "--data_mode",
                            "Non-IID",
                            "--client_dir",
                            "clients_non_iid",
                            "--save",
                            f"fedprox_mu{mu}_r{rounds}_e{local_epochs}_s{seed}.pth",
                            "--config_path",
                            str(config_path),
                            "--checkpoint_dir",
                            str(PROJECT_ROOT / "results" / "checkpoints"),
                        ],
                        PROJECT_ROOT / "Federated Learning" / "CNN_FedProx",
                    )


def run_phase4():
    run_cmd([PYTHON_BIN, "aggregate_results.py"], PROJECT_ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run experiments phase-by-phase")
    parser.add_argument("--phase", type=int, required=True, choices=[1, 2, 3, 4], help="Phase number to execute")
    parser.add_argument("--config", type=str, default="experiment_config.json", help="Experiment config path")
    args = parser.parse_args()

    config_path = PROJECT_ROOT / args.config
    if args.phase == 1:
        run_phase1(config_path)
    elif args.phase == 2:
        run_phase2(config_path)
    elif args.phase == 3:
        run_phase3(config_path)
    elif args.phase == 4:
        run_phase4()


if __name__ == "__main__":
    main()
