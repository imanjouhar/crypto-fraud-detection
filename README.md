<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-0D9488?style=for-the-badge&logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/XGBoost-Classifier-0F1B2D?style=for-the-badge&logoColor=white">
  <img src="https://img.shields.io/badge/Flask-REST_API-14B8A6?style=for-the-badge&logo=flask&logoColor=white">
  <img src="https://img.shields.io/badge/MLflow-Tracking-0194E2?style=for-the-badge&logo=mlflow&logoColor=white">
  <img src="https://img.shields.io/badge/Docker-Container-2496ED?style=for-the-badge&logo=docker&logoColor=white">
  <img src="https://img.shields.io/badge/GitHub_Actions-CI/CD-2088FF?style=for-the-badge&logo=githubactions&logoColor=white">
</p>

<h1 align="center">Anti-Money Laundering Detection<br>on Bitcoin Transactions</h1>

<p align="center">
  <strong>MLOps Pipeline — From Model to Production</strong><br>
  <em>DLBDSMTP01 · IU International University of Applied Sciences</em>
</p>

<p align="center">
  <strong>Iman Jouhar</strong><br>
  Task 3 — Fraud Detection (Spotlight: MLOps)
</p>

---

## Overview

This project builds a machine learning system that identifies illicit transactions on the Bitcoin blockchain. It covers every stage of the production lifecycle: training, serving, monitoring, drift detection, and automated retraining.

The system is trained on the [Elliptic Bitcoin Dataset](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set), a labeled graph of real Bitcoin transactions published by Elliptic, a blockchain analytics company. The dataset contains over 200,000 transactions, of which approximately 46,000 are labeled as either licit or illicit. Each transaction is described by 165 anonymized features derived from the blockchain.

The core challenge in AML (Anti-Money Laundering) detection is the class imbalance: only about 10% of labeled transactions are illicit. The model addresses this using XGBoost with weighted classes and PCA-based dimensionality reduction.

---

## System Architecture

The prediction pipeline and the MLOps layer operate as two connected systems:

```
┌────────────────┐     ┌───────────────┐     ┌────────────┐     ┌───────────────────┐
│  Bitcoin       │────▶│  Scaler + PCA │────▶│  XGBoost   │────▶│  Flask REST API   │
│  Transaction   │     │  (165 → 30)   │     │  Classifier│     │  /predict         │
└────────────────┘     └───────────────┘     └──────┬─────┘     └────────┬──────────┘
                                                    │                     │
                                                    │             ┌───────▼──────────┐
                                                    │             │  Risk Score      │
                                                    │             │  LOW / MED / HIGH│
                                                    │             └──────────────────┘
                    ┌───────────────────────────────┘
                    │
       ┌────────────▼────────────┐   ┌─────────────────┐   ┌──────────────────┐
       │  Drift Detection       │──▶│  GitHub Actions  │──▶│  MLflow          │
       │  (KS Test)             │   │  Retrain + Deploy│   │  Experiment Log  │
       └─────────────────────────┘   └─────────────────┘   └──────────────────┘
```

---

## How to Run

### 1. Setup

```bash
git clone https://github.com/imanjouhar/aml-fraud-detection.git
cd aml-fraud-detection
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Dataset

Download the Elliptic Bitcoin dataset from Kaggle and place the three CSV files in a `data/` folder:

> https://www.kaggle.com/datasets/ellipticco/elliptic-data-set

```
data/
├── elliptic_txs_features.csv
├── elliptic_txs_classes.csv
└── elliptic_txs_edgelist.csv
```

### 3. Run the pipeline

```bash
python main.py              # Full pipeline: train → drift baseline → 12-month simulation → charts
python main.py train        # Train the model
python main.py api          # Start the REST API on port 5000
python main.py simulate     # Run the 12-month drift simulation
python main.py visualize    # Generate all result charts
python main.py dashboard    # Open interactive HTML dashboard in browser
python main.py monitor      # Open MLflow tracking UI on port 5001
python main.py demo         # Quick train + single prediction
```

---

## Project Structure

```
├── main.py                          All pipeline logic in one file
├── visualize.py                     Chart generation and HTML dashboard
├── requirements.txt                 Python dependencies
├── Dockerfile                       Container packaging for deployment
├── .github/workflows/retrain.yml    GitHub Actions retraining pipeline
├── models/                          Trained model artifacts (.pkl)
├── data/                            Elliptic CSV files (not tracked in git)
├── visualizations/                  Generated PNG charts (created at runtime)
└── README.md
```

---

## Dataset Details

| Property | Value |
|----------|-------|
| Source | Elliptic via Kaggle |
| Type | Real Bitcoin blockchain transactions |
| Total transactions | 203,769 |
| Labeled transactions | ~46,500 |
| Illicit (class 1) | 4,545 (~9.8% of labeled) |
| Licit (class 2) | 42,019 (~90.2% of labeled) |
| Features per transaction | 165 anonymized |
| Time steps | 49 (each covering ~2 weeks) |
| Graph edges | 234,355 directed payment flows |

The 165 features are split into two groups: 94 local features derived directly from the transaction, and 71 aggregated features computed from a transaction's one-hop neighborhood in the graph.

---

## Model

The classifier is XGBoost with PCA preprocessing:

| Component | Detail |
|-----------|--------|
| Preprocessing | StandardScaler → PCA (165 → 30 components, ~85% variance retained) |
| Additional features | time_step, in_degree, out_degree (from edge list) |
| Classifier | XGBClassifier, 200 trees, max depth 6 |
| Class imbalance | Handled via `scale_pos_weight` (~9:1 ratio) |
| Evaluation metric | AUCPR (area under precision-recall curve) |
| Split | 80/20 stratified |

---

## REST API

```bash
python main.py api
```

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| GET | `/health` | No | Returns API status |
| POST | `/predict` | API Key | Scores one transaction |
| POST | `/predict/batch` | API Key | Scores multiple transactions |

All prediction endpoints require an `X-API-Key` header. Unauthorized requests are rejected with HTTP 401.

Example:

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: aml-secret-key-2024" \
  -d '{"feat_0": 0.5, "feat_1": -0.3, "time_step": 25, "in_degree": 3, "out_degree": 2}'
```

Response:

```json
{
  "is_illicit": 1,
  "probability": 0.7823,
  "risk_level": "HIGH"
}
```

---

## Drift Detection

The system monitors for data drift using the Kolmogorov-Smirnov two-sample test. Each incoming batch of transactions is compared against the training distribution.

- Per feature: if p-value < 0.05, that feature is flagged as drifted
- Overall: if 30% or more of monitored features drift, retraining is triggered

This is important in AML because criminal behavior evolves over time. A model trained on historical patterns will degrade as new laundering techniques emerge.

---

## 12-Month Simulation

The 49 time steps in the Elliptic dataset are grouped into 12 monthly windows. Progressive noise is applied to simulate changing conditions:

| Period | Drift applied |
|--------|--------------|
| Months 1–3 | None (stable baseline) |
| Months 4–6 | Mild noise injection |
| Months 7–9 | Feature scaling + moderate noise |
| Months 10–12 | Heavy scaling + label perturbation |

The simulation validates that the pipeline correctly detects drift and triggers retraining at the right moments.

---

## CI/CD Pipeline

GitHub Actions automates the retraining cycle with three triggers:

1. **Scheduled** — runs on the 1st of each month at 02:00
2. **Data push** — triggers when new files are added to `data/`
3. **Manual** — can be dispatched from the Actions tab

The pipeline runs: drift check → retrain (if needed) → validation → Docker rebuild → API health check.

---

## Monitoring

MLflow tracks every training run and monthly evaluation:

- Hyperparameters (estimators, depth, PCA components)
- Metrics (precision, recall, F1, ROC-AUC)
- Model artifacts (stored and versioned)

Launch the dashboard:

```bash
python main.py monitor
```

Then open http://localhost:5001 in a browser.

---

## Docker

```bash
docker build -t aml-api .
docker run -p 5000:5000 -e API_KEY=your-key aml-api
```

---

## Visualizations

Running `python main.py visualize` generates:

- Class distribution chart
- PCA 2D scatter (licit vs illicit clusters)
- Confusion matrix heatmap
- ROC curve with AUC
- Precision-recall curve
- Feature importance ranking
- Time step analysis (illicit rate over time)
- Drift timeline from simulation
- Metrics summary card

Running `python main.py dashboard` opens all charts in an interactive HTML page.

---

## Tools

| | |
|---|---|
| Language | Python 3.11 |
| ML | XGBoost, scikit-learn (PCA, StandardScaler) |
| API | Flask |
| Monitoring | MLflow |
| Drift detection | SciPy (KS test) |
| CI/CD | GitHub Actions |
| Container | Docker |
| Charts | Matplotlib |

---

## Course

| | |
|---|---|
| Course | DLBDSMTP01 — Project: From Model to Production |
| University | IU International University of Applied Sciences |
| Task | 3 — Fraud Detection (Spotlight: MLOps) |
| Author | Iman Jouhar |

---

## License

MIT — see [LICENSE](LICENSE) for details.
