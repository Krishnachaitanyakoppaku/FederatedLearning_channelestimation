# IEEE-Style Comparative Evaluation Report

## Abstract
This report presents an IEEE-style comparative evaluation of the proposed wireless channel estimation framework against selected prior federated learning (FL) literature. The primary quantitative external benchmark is the domain-matched work in `papers/2008.10846v2.pdf`, while `papers/1602.05629v4.pdf` and `papers/1812.06127v5.pdf` are used as algorithmic baselines for FedAvg and FedProx behavior. Internal results are sourced from structured multi-run experiments and summarized with mean, standard deviation, and best-case normalized mean squared error (NMSE, dB).

## I. Scope and Benchmark Finalization
- **Primary quantitative benchmark:** `papers/2008.10846v2.pdf` (FL for channel estimation in conventional and RIS-assisted massive MIMO).
- **Algorithmic benchmark references:** `papers/1602.05629v4.pdf` (FedAvg), `papers/1812.06127v5.pdf` (FedProx).
- **Context-only references:** `papers/1906.06629v2.pdf`, `papers/1912.04977v3.pdf` (not used for direct numeric NMSE comparison).

## II. Experimental Source Data (This Work)
- **Aggregate comparison table:** `results/summary/final_comparison_table.csv`
- **Configuration-level statistics:** `results/summary/main_comparison_table.csv`
- **Best configurations:** `results/summary/paper_table_1_best_configs.csv`
- **Statistical summary:** `results/summary/paper_table_2_statistical_summary.csv`
- **Communication summary:** `results/summary/paper_table_4_communication_cost.csv`

## III. Main Quantitative Comparison

### TABLE I
### Main Quantitative Comparison (NMSE in dB) With Match-Level Annotation

| Study | Task | Model/Setting | Data Mode | SNR (dB) | CL NMSE (dB) | FL NMSE (dB) | FL-CL Gap (dB) | Communication Indicator | Match Level | Source |
|---|---|---|---|---:|---:|---:|---:|---|---|---|
| This work | Channel estimation | CNN (Centralized) | N/A | 10 | -19.4227 (mean), -19.6215 (best) | N/A | N/A | 0 FL comm events | Matched (self) | `results/summary/final_comparison_table.csv` |
| This work | Channel estimation | CNN + FedAvg | IID | 10 | -19.4227 | -19.1190 (mean), -20.0381 (best) | +0.3036 (mean) | R50,E5 => 250 events | Matched (self) | `results/summary/paper_table_2_statistical_summary.csv` |
| This work | Channel estimation | CNN + FedAvg | Non-IID | 10 | -19.4227 | -18.8483 (mean), -19.7338 (best) | +0.5743 (mean) | FL setting-dependent | Matched (self) | `results/summary/paper_table_2_statistical_summary.csv` |
| This work | Channel estimation | CNN + FedProx | Non-IID | 10 | -19.4227 | -18.5683 (mean), -19.6479 (best) | +0.8544 (mean) | FL setting-dependent | Matched (self) | `results/summary/paper_table_2_statistical_summary.csv` |
| Elbir and Coleri | Channel estimation (conventional and RIS-assisted massive MIMO) | CNN + FL | Paper-defined | Paper-defined | Paper-defined | Paper-defined | Paper-defined | ~16x lower overhead than CL (reported) | Partially matched | `papers/2008.10846v2.pdf` |

**Note:** External numeric cells are marked "Paper-defined" and should be filled from the exact tables/figures in `papers/2008.10846v2.pdf` to ensure citation-accurate reporting.

## IV. Algorithm-Level Consistency With Foundational FL Literature

### TABLE II
### Algorithmic Consistency Check (Trend-Level, Not Absolute NMSE Equality)

| Algorithm | Foundational Reference | Literature Claim | Observation in This Work | Consistency |
|---|---|---|---|---|
| FedAvg | `papers/1602.05629v4.pdf` | Communication-efficient decentralized optimization; sensitive to non-IID heterogeneity | FedAvg-IID mean NMSE (-19.1190) outperforms FedAvg-Non-IID mean NMSE (-18.8483) | Yes |
| FedProx | `papers/1812.06127v5.pdf` | Improved robustness under heterogeneity via proximal term | FedProx evaluated with `mu` sweep under Non-IID regime and compared to FedAvg trends | Partially (trend validated; absolute dominance not universal) |

## V. Fairness and Comparability Matrix

### TABLE III
### Apples-to-Apples Assessment for External Comparison

| Dimension | This Work | `papers/2008.10846v2.pdf` | Status |
|---|---|---|---|
| Problem class | Wireless channel estimation | Wireless channel estimation | Matched |
| Learning paradigm | Centralized + FL (FedAvg, FedProx) | Centralized + FL | Matched |
| Reported metric | NMSE (dB) | Estimation error/NMSE-style reporting | Verify exact definition in paper |
| SNR protocol | 10 dB in current summary set | Multi-SNR in paper | Partially matched |
| Channel/system assumptions | Project-specific dataset and setup | Conventional and RIS-assisted massive MIMO setup | Partially matched |
| Model class | CNN, ResUNet | CNN | Partially matched |
| FL protocol details | Multi-seed grid over rounds/local epochs/mu | Paper-defined FL settings | Partially matched |

## VI. Communication-Efficiency Snapshot (Internal)

### TABLE IV
### Communication Cost Proxy From This Work

| Configuration | Rounds (R) | Local Epochs (E) | Communication Events (R x Clients) | NMSE (dB) | Efficiency Label |
|---|---:|---:|---:|---:|---|
| FedAvg-IID (R10, E1) | 10 | 1 | 50 | -17.7319 | Low |
| FedAvg-IID (R30, E3) | 30 | 3 | 150 | -19.5375 | Medium |
| FedAvg-IID (R50, E5) | 50 | 5 | 250 | -20.0190 | High |
| Centralized (20 epochs) | N/A | 20 | 0 (no FL rounds) | -19.4227 | N/A |

Source: `results/summary/paper_table_4_communication_cost.csv`

## VII. Key Findings
- The strongest internal NMSE result among listed CNN settings is FedAvg-IID at `R=50, E=5` with `-20.0190 dB`.
- Mean performance ranking at 10 dB is: Centralized (`-19.4227`) > FedAvg-IID (`-19.1190`) > FedAvg-Non-IID (`-18.8483`) > FedProx-Non-IID (`-18.5683`).
- Non-IID data split introduces measurable degradation relative to IID for FedAvg in the current setup.
- External quantitative comparison should prioritize `papers/2008.10846v2.pdf` and explicitly disclose partially matched assumptions.

## VIII. Recommended IEEE-Style Reporting Language
"The proposed framework is benchmarked against established FL references and a domain-specific channel estimation FL baseline. In the 10 dB setting, centralized CNN achieves -19.4227 dB mean NMSE, while FedAvg-IID achieves -19.1190 dB mean NMSE (best: -20.0190 dB), and non-IID federated settings incur additional degradation. External comparison with Elbir and Coleri is reported with explicit comparability annotations due to differences in system assumptions and SNR protocol."

## IX. References (Selected)
1. H. B. McMahan *et al.*, "Communication-Efficient Learning of Deep Networks from Decentralized Data," AISTATS, 2017. (`papers/1602.05629v4.pdf`)
2. T. Li *et al.*, "Federated Optimization in Heterogeneous Networks," MLSys, 2020. (`papers/1812.06127v5.pdf`)
3. A. M. Elbir and S. Coleri, "Federated Learning for Channel Estimation in Conventional and RIS-Assisted Massive MIMO," IEEE TWC, 2021. (`papers/2008.10846v2.pdf`)
4. A. Ghosh *et al.*, "Robust Federated Learning in a Heterogeneous Environment," 2019. (`papers/1906.06629v2.pdf`)
5. P. Kairouz *et al.*, "Advances and Open Problems in Federated Learning," 2021. (`papers/1912.04977v3.pdf`)
