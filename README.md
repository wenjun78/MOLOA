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
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place the Freddie Mac SFLLD sample files in a data folder:
#      MOLOA_data/sample_orig_2022.txt
#      MOLOA_data/sample_svcg_2022.txt

# 3. Point the notebook at that folder (or set the env var):
export MOLOA_DATA_DIR=./MOLOA_data

# 4. Open and run the notebook end-to-end
jupyter notebook notebook/MOLOA_full_pipeline.ipynb
```

On Google Colab the notebook mounts Drive automatically — just put the data in `MyDrive/MOLOA_data/`.

---

## What problem this solves

Three structural gaps in conventional mortgage credit modelling:

1. **Black-box classifiers** predict default but cannot say *which named risk dimension* drove the prediction.
2. **Siloed lifecycle modelling** treats origination, monitoring, and intervention as disconnected problems, discarding the origination signal.
3. **Confounded modification analysis** defends loss-mitigation decisions through selection-biased observational comparisons rather than causal estimates.

MOLOA addresses each with one phase.

---

## Architecture

| Phase | Trigger | What it does | Key components |
|-------|---------|--------------|----------------|
| **Phase 1 — Origination** | Month 0 | Decomposed credit decisioning across 5 domain agents | Learned Agent Router (LAR), Conflict Arbitration Network (CAN), Adaptive Escalation Logic (AEL) |
| **Phase 2 — Early Monitoring** | Month 6 | Survival modelling with lifecycle carry-forward | Cox proportional-hazards model, KalmanHD trajectory features |
| **Phase 3 — Distress Intervention** | First 60+DPD | Causal effect of loan modification | IPTW Average Treatment Effect, T-learner CATE |

The Phase 1 routed probability is carried forward as an explicit feature into Phase 2 — the lifecycle hand-off, not just a boundary.

---

## Headline results

Evaluated on 50,000 loans, 1.82% default prevalence (1:54 class imbalance), 80/20 stratified split with 5-fold cross-validation.

**Phase 1 — ten-model benchmark.** MOLOA leads on the imbalance-sensitive metrics:

| Metric | MOLOA | Notes |
|--------|-------|-------|
| F1-macro | **0.5535** | MOLOA leads |
| AUPRC | **0.0793** | MOLOA leads |
| Default-class F1 | **0.1279** | MOLOA leads |
| ROC-AUC (CV) | 0.7408 | Logistic Regression leads at 0.7928 — reported, not smoothed over |

- CAN disagreement detection: AUC **0.9819**
- AEL cost-optimal escalation: **21.8%** of applications

**Phase 2 — survival.** Cox concordance **0.9707 ± 0.0026** (5-fold CV). The Phase 1 LAR probability carries forward at coefficient +2.180, p < 6×10⁻²⁶⁰. KalmanHD residual significant at p < 1.3×10⁻⁹. Phases 1 and 2 agree on 96.91% of month-6 loans.

**Phase 3 — causal.** IPTW Average Treatment Effect of modification on six-month cure: **+25.3 pp** (95% bootstrap CI [+17.3, +31.7]), propensity model AUC 0.94. T-learner shows substantial heterogeneity (41.6% of loans have a CI excluding zero).

Inference: 3.60 ms per loan, ~91,000 loans/sec on a Colab T4.

---

## Contributions

1. **Three-phase coordinated multi-agent architecture** novel to residential mortgage modelling.
2. **Learned Agent Router** with entropy-regularised attention that keeps the gate distribution diverse instead of collapsing onto the dominant agent.
3. **Ten-model SFLLD 2022 benchmark** reporting both wins and honest non-wins.
4. **First reviewed application of KalmanHD** (hyperdimensional computing with Kalman-style state updates) to mortgage trajectory features.

---

## Getting the data

The Freddie Mac SFLLD is **not redistributable**, so the two raw files are not in this repository. You must obtain them yourself (free):

1. Go to the [Freddie Mac Single-Family Loan-Level Dataset page](https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset).
2. Register for a free account and accept the licence agreement.
3. Download the **2022 sample** release.
4. Place the two text files in your data folder (`MOLOA_data/` by default):

   ```
   MOLOA_data/sample_orig_2022.txt    # 50,000 origination records
   MOLOA_data/sample_svcg_2022.txt    # 1,788,639 monthly performance records
   ```

The notebook reads these from `$MOLOA_DATA_DIR` and builds `moloa_master.parquet` on the first run. Nothing is downloaded automatically — the files must already be in place.

## Scoring a single application

Once the notebook has produced `phase1_models.pkl`, you can score an applicant from the command line without re-running the notebook:

```bash
# Score the built-in example applicant
python predict.py --demo

# Score from your own JSON file of feature values
python predict.py --input applicant.json
```

Output is a JSON object with the routed default probability, the approve/deny decision, each agent's probability, and the router's per-agent gate weights — so you can see *which* risk dimension drove the decision:

```json
{
  "default_probability": 0.0412,
  "decision_threshold": 0.46,
  "decision": "APPROVE",
  "agent_probabilities": { "CreditAgent": 0.03, "CollateralAgent": 0.08, ... },
  "router_gate_weights":  { "CreditAgent": 0.61, "CollateralAgent": 0.14, ... }
}
```

## Repository structure

```
.
├── README.md
├── requirements.txt                  # Python dependencies
├── .gitignore
├── LICENSE
├── predict.py                        # CLI: score a single application
├── notebook/
│   └── MOLOA_full_pipeline.ipynb     # End-to-end: Phases 0+1+2+3
├── results/
│   ├── phase1_benchmark_results.csv  # 10-model benchmark (single split)
│   ├── phase1_benchmark_cv_folds.csv # 5-fold cross-validation AUC
│   ├── phase1_inference_speed.csv    # Per-loan & batch latency
│   ├── hdc_benchmark_results.json    # KalmanHD trajectory benchmark
│   └── phase1_benchmark_plots.png    # Benchmark figure
└── demo/
    └── moloa_demo.html               # Interactive HTML demo
```

---

## Reproducing the results

The pipeline runs end-to-end in the notebook on Google Colab (T4 GPU recommended).

1. Obtain the Freddie Mac SFLLD 2022 sample (`sample_orig_2022.txt`, `sample_svcg_2022.txt`) from the [Freddie Mac Single-Family Loan-Level Dataset](https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset).
2. Open `notebook/MOLOA_full_pipeline.ipynb` in Colab.
3. Run all cells. The notebook builds the master dataset, trains all three phases, and writes results to your Drive.

> The raw SFLLD files, trained model pickles (`phase*_models.pkl`), and prediction parquets are **not included** in this repository — the dataset is not redistributable under Freddie Mac's terms, and the model artifacts are regenerated by running the notebook.

---

## Dataset

[Freddie Mac Single-Family Loan-Level Dataset](https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset), 2022 vintage — the canonical public academic benchmark for residential mortgage credit risk. 50,000-loan sample, 28 origination features, 24-month observation window.

---

## License

This research project is provided for academic and educational purposes. The Freddie Mac SFLLD is subject to Freddie Mac's own terms of use.
