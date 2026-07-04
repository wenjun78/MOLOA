<div align="center">

# MOLOA

### Multi-Agent Orchestration for Loan Origination and Aftercare

*A three-phase multi-agent AI system for residential mortgage credit-risk decisioning*

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/status-research--prototype-lightgrey.svg)

**WQF7023 Research Project · Master of Artificial Intelligence · Universiti Malaya**

Author: Lau Wen Jun · Supervisor: Prof Loo Chu Kiong

</div>

---

MOLOA follows a residential mortgage through its full lifecycle: **origination** at month 0, **early monitoring** at month 6, and **distress intervention** at the first 60+ days-past-due event. It is evaluated on the Freddie Mac Single-Family Loan-Level Dataset (SFLLD) 2022.

## Quick start

```bash
# 1. Clone the repository
git clone https://github.com/wenjun78/MOLOA.git
cd MOLOA

# 2. Install dependencies
pip install -r requirements.txt

# 3. Score two example applicants across all three phases (no data files needed)
python predict.py --demo

# 4. Trace one real loan through all three phases (reads the committed prediction files)
python demo_full.py

# 5. Open the interactive demo — locally or at the live link:
#    https://wenjun78.github.io/MOLOA/demo/moloa_demo.html
```

All three demos run out of the box after cloning — no downloads or GPU required.

---

## What problem this solves

Three structural gaps in conventional mortgage credit modelling:

1. **Black-box classifiers** predict default but cannot say *which named risk dimension* drove the prediction.
2. **Siloed lifecycle modelling** treats origination, monitoring, and intervention as disconnected problems, discarding the origination signal.
3. **Confounded modification analysis** defends loss-mitigation decisions through selection-biased observational comparisons rather than causal estimates.

MOLOA addresses each with one phase.

## Architecture

| Phase | Trigger | What it does | Key components |
|-------|---------|--------------|----------------|
| **Phase 1 — Origination** | Month 0 | Decomposed credit decisioning across 5 domain agents | Learned Agent Router (LAR), Conflict Arbitration Network (CAN), Adaptive Escalation Logic (AEL) |
| **Phase 2 — Early Monitoring** | Month 6 | Survival modelling with lifecycle carry-forward | Cox proportional-hazards model, KalmanHD trajectory features |
| **Phase 3 — Distress Intervention** | First 60+DPD | Causal effect of loan modification | IPTW Average Treatment Effect, T-learner CATE |

The Phase 1 routed probability is carried forward as an explicit feature into
Phase 2 — the lifecycle hand-off, not just a boundary.

## Headline results

Evaluated on 50,000 loans, 1.82% default prevalence (1:54 class imbalance),
80/20 stratified split with 5-fold cross-validation.

**Phase 1 — ten-model benchmark.** MOLOA leads on the imbalance-sensitive metrics:

| Metric | MOLOA | Notes |
|--------|-------|-------|
| F1-macro | **0.5535** | MOLOA leads |
| AUPRC | **0.0793** | MOLOA leads |
| Default-class F1 | **0.1279** | MOLOA leads |
| ROC-AUC (CV) | 0.7408 | Logistic Regression leads at 0.7928 — reported, not smoothed over |

- CAN disagreement detection: AUC **0.9819**
- AEL cost-optimal escalation: **21.8%** of applications

**Phase 2 — survival.** Cox concordance **0.9707 ± 0.0026** (5-fold CV). The Phase 1
LAR probability carries forward at coefficient +2.180, p < 6×10⁻²⁶⁰. KalmanHD
residual significant at p < 1.3×10⁻⁹. Phases 1 and 2 agree on 96.91% of month-6 loans.

**Phase 3 — causal.** IPTW Average Treatment Effect of modification on six-month
cure: **+25.3 pp** (95% bootstrap CI [+17.3, +31.7]), propensity model AUC 0.94.
T-learner shows substantial heterogeneity (41.6% of loans have a CI excluding zero).

Inference: 3.60 ms per loan, ~91,000 loans/sec on a Colab T4.

## Contributions

1. **Three-phase coordinated multi-agent architecture** novel to residential mortgage modelling.
2. **Learned Agent Router** with entropy-regularised attention that keeps the gate distribution diverse instead of collapsing onto the dominant agent.
3. **Ten-model SFLLD 2022 benchmark** reporting both wins and honest non-wins.
4. **First reviewed application of KalmanHD** (hyperdimensional computing with Kalman-style state updates) to mortgage trajectory features.

---

## Demos

Three ways to see MOLOA in action. All run without a GPU and are robust to
library-version differences.

### 1. `predict.py` — score an applicant across all three phases

Scores a hypothetical applicant through Phase 1 → 2 → 3 using a lightweight,
self-contained approximation of the trained models. Needs no data files.

```bash
# Built-in strong vs weak applicants
python predict.py --demo

# From a JSON file
python predict.py --input applicant_strong.json

# Directly from the command line
python predict.py --fico 650 --dti 40 --ltv 90 --rate 7.5 --upb 300000 --months-late 2
```

> **Note.** `predict.py` uses a calibrated *approximation* of the trained models
> (the same approach as the "What-if" tab in the HTML demo below), so it can score
> hypothetical applicants interactively. The actual trained-model outputs are the
> prediction files in `demo/` that `demo_full.py` reads, and the results reported
> in the benchmark.

### 2. `demo_full.py` — trace one real loan through its lifecycle

Reads the committed prediction files in `demo/` and follows a single real loan
across all three phases: approved at origination, risk updated at month 6, and an
intervention recommendation at distress.

```bash
# Auto-pick an interesting loan present in all three phases
python demo_full.py

# List the top demo candidates (shows the total count too)
python demo_full.py --list

# List every loan present in all three phases
python demo_full.py --all

# Trace a specific loan
python demo_full.py --loan F22Q20008051
```

### 3. `demo/moloa_demo.html` — interactive browser demo

A self-contained interactive walkthrough (five agents, architecture, real loans,
and a live "What-if" scorer). Just open the file — no server, no install.

**Live version:** https://wenjun78.github.io/MOLOA/demo/moloa_demo.html

In the **What-if** tab, drag the **FICO** slider to flip the decision between
APPROVE and DENY, and set **M6 Max Delinquency** to 2 or higher to trigger the
Phase 3 distress path.

---

## Reproducing the full pipeline

The demos above run on their own. To rebuild everything from the raw data, run the
notebook end-to-end on Google Colab (T4 GPU recommended).

The Freddie Mac SFLLD is **not redistributable**, so the raw files are not in this
repository. Obtain them yourself (free):

1. Go to the [Freddie Mac Single-Family Loan-Level Dataset page](https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset).
2. Register for a free account and accept the licence agreement.
3. Download the **2022 sample** release.
4. Place the two text files in a data folder (`MOLOA_data/` by default):

   ```
   MOLOA_data/sample_orig_2022.txt    # 50,000 origination records
   MOLOA_data/sample_svcg_2022.txt    # 1,788,639 monthly performance records
   ```

5. Open `notebook/MOLOA_full_pipeline.ipynb` in Colab, point it at that folder
   (env var `MOLOA_DATA_DIR`, defaults to `./MOLOA_data`), and run all cells. The
   notebook builds the master dataset, trains all three phases, and writes the
   model and prediction files.

> The raw SFLLD files and the trained model pickles (`phase*_models.pkl`) are not
> included — the dataset is not redistributable under Freddie Mac's terms, and the
> models are regenerated by running the notebook. The three small
> `phase*_predictions.parquet` files **are** included in `demo/` so the demos run
> without any download.

---

## Repository structure

```
.
├── README.md
├── requirements.txt                  # Python dependencies
├── .gitignore
├── LICENSE
├── predict.py                        # Demo 1: score an applicant across all 3 phases
├── demo_full.py                      # Demo 2: trace one real loan across all 3 phases
├── applicant_strong.json             # Example low-risk applicant for predict.py
├── applicant_weak.json               # Example high-risk applicant for predict.py
├── notebook/
│   └── MOLOA_full_pipeline.ipynb     # End-to-end: Phases 0+1+2+3
├── results/
│   ├── phase1_benchmark_results.csv  # 10-model benchmark (single split)
│   ├── phase1_benchmark_cv_folds.csv # 5-fold cross-validation AUC
│   ├── phase1_inference_speed.csv    # Per-loan & batch latency
│   ├── hdc_benchmark_results.json    # KalmanHD trajectory benchmark
│   └── phase1_benchmark_plots.png    # Benchmark figure
└── demo/
    ├── moloa_demo.html               # Demo 3: interactive browser walkthrough
    ├── phase1_predictions.parquet    # Phase 1 predictions (read by demo_full.py)
    ├── phase2_predictions.parquet    # Phase 2 predictions
    └── phase3_predictions.parquet    # Phase 3 predictions
```

---

## Dataset

[Freddie Mac Single-Family Loan-Level Dataset](https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset),
2022 vintage — the canonical public academic benchmark for residential mortgage
credit risk. 50,000-loan sample, 28 origination features, 24-month observation window.

## License

Released under the MIT License — see [LICENSE](LICENSE). The Freddie Mac SFLLD is
subject to Freddie Mac's own terms of use and is not redistributed here.
