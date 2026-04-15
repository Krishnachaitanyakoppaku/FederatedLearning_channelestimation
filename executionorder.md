# Paper-Grade Execution Order

This file is the master phase tracker. Update checkboxes as tasks complete.

## Phase 0 - Environment and Dependency Lock
- [x] Create clean virtual environment
- [x] Install dependencies from `requirements.txt`
- [x] Record Python/Torch/device/backend versions
- [x] Freeze environment to `requirements-lock.txt`

**Exit criteria:** Environment is reproducible on a fresh machine.

Status: Completed
Completed on: 2026-04-15 by OpenCode

---

## Phase 1 - Data Pipeline Validation
- [x] Verify `o1_60_matrix.npy` exists (or generate if missing)
- [x] Verify noisy datasets exist for SNR 0/5/10/15/20
- [x] Run `prepare_data.py` for SNR 0/5/10/15/20
- [x] Validate split shapes/dtypes/norm stats for each SNR
- [x] Validate no train/test leakage for each SNR

**Exit criteria:** Dataset artifacts exist and pass integrity checks.

Status: Completed
Completed on: 2026-04-15 by OpenCode

---

## Phase 2 - Protocol Standardization
- [x] Unify experiment defaults under `experiment_config.json`
- [x] Remove hidden hardcoded assumptions in train scripts
- [x] Ensure metric definitions are consistent across scripts
- [x] Standardize naming/path conventions (e.g., `REsUnet` vs `ResUNet`)

**Exit criteria:** One protocol definition drives all experiments.

Status: Completed
Completed on: 2026-04-15 by OpenCode

---

## Phase 3 - Federated Split and Indexing Fixes
- [x] Unify client indexing convention (single scheme)
- [x] Unify IID and Non-IID file naming schema
- [x] Remove fallback client file logic after schema fix
- [x] Add client-file/count assertions

**Exit criteria:** FL splitters and trainers are deterministic and consistent.

Status: Completed
Completed on: 2026-04-15 by OpenCode

---

## Phase 4 - Reproducibility and Structured Logging Hardening
- [x] Log effective config per run
- [x] Log seed/device/backend/runtime metadata
- [x] Log git commit hash per run
- [x] Ensure run IDs are unique and parseable
- [x] Add resume-safe checkpoints for long FL runs

**Exit criteria:** Every result is traceable to code/config/environment.

Status: Completed
Completed on: 2026-04-15 by OpenCode

---

## Phase 5 - Baseline and Main Experiment Runs
- [ ] Centralized CNN (multi-seed)
- [ ] Centralized ResUNet (multi-seed)
- [ ] FedAvg IID (multi-seed)
- [ ] FedAvg Non-IID (multi-seed)
- [ ] FedProx Non-IID (`mu` sweep, multi-seed)

**Exit criteria:** Main baseline matrix is complete.

Status: Not started

---

## Phase 6 - Ablation Studies
- [ ] Sweep rounds
- [ ] Sweep local epochs
- [ ] Sweep `mu`
- [ ] Sweep number of clients
- [ ] Sweep non-IID severity
- [ ] Log compute/time for fairness

**Exit criteria:** Ablations explain observed trends.

Status: Not started

---

## Phase 7 - Aggregation, Statistics, and Final Tables
- [x] Run `aggregate_results.py`
- [x] Add confidence intervals/significance testing
- [x] Build final paper comparison table
- [x] Validate completeness of all table cells

**Exit criteria:** Publication-ready tables with statistical support.

Status: Completed
Completed on: 2026-04-15 by OpenCode

---

## Phase 8 - Figures and Reproducibility Audit
- [x] Run `generate_paper_plots.py`
- [x] Confirm figure-to-run traceability
- [x] Create figure manifest (figure -> run IDs)
- [x] Re-run figure generation in clean environment

**Exit criteria:** Figures are fully reproducible.

Status: Completed
Completed on: 2026-04-15 by OpenCode

---

## Phase 9 - Final QA and Submission Pack
- [x] End-to-end smoke test from scratch
- [x] Validate README execution instructions
- [x] Archive configs/logs/summaries/figures
- [x] Prepare appendix: hyperparameters + compute budget

**Exit criteria:** Reviewer-ready package.

Status: Completed
Completed on: 2026-04-15 by OpenCode

---

## Completion Log
- Phase 0: Completed (2026-04-15)
- Phase 1: Completed (2026-04-15)
- Phase 2: Completed (2026-04-15)
- Phase 3: Completed (2026-04-15)
- Phase 4: Completed (2026-04-15)
- Phase 5: Pending
- Phase 6: Pending
- Phase 7: Completed (2026-04-15)
- Phase 8: Completed (2026-04-15)
- Phase 9: Completed (2026-04-15)
