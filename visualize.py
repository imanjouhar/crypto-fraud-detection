"""
Interactive EDA Dashboard – AML Bitcoin Fraud Detection
Iman Jouhar | DLBDSMTP01
"""

import os, json, glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    precision_recall_curve, classification_report,
    precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import joblib

OUTPUT_DIR = "visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Static PNG Charts ──────────────────────────────────────

def plot_class_distribution(y):
    """Bar chart of licit vs illicit transaction counts."""
    counts = y.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(["Licit", "Illicit"], counts.values, color=["#0D9488", "#EF4444"], width=0.5)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                f"{val:,}\n({val/len(y)*100:.1f}%)", ha="center", fontsize=12, fontweight="bold")
    ax.set_ylabel("Transactions"); ax.set_title("Class Distribution")
    ax.set_ylim(0, counts.max() * 1.15); plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "01_class_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  saved {path}")

def plot_confusion_matrix(y_true, y_pred):
    """Confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i,j] > cm.max()/2 else "#0F1B2D"
            ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center", fontsize=18, fontweight="bold", color=color)
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Licit","Illicit"]); ax.set_yticklabels(["Licit","Illicit"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion Matrix")
    plt.colorbar(im, ax=ax, shrink=0.8); plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "02_confusion_matrix.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  saved {path}")

def plot_roc(y_true, y_proba):
    """ROC curve with AUC."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#0D9488", linewidth=2.5, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0,1],[0,1],"k--",alpha=0.4); ax.fill_between(fpr, tpr, alpha=0.1, color="#0D9488")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("ROC Curve"); ax.legend()
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "03_roc_curve.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  saved {path}")

def plot_pr(y_true, y_proba):
    """Precision-recall curve."""
    prec, rec, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(rec, prec)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(rec, prec, color="#EF4444", linewidth=2.5, label=f"PR-AUC = {pr_auc:.4f}")
    ax.fill_between(rec, prec, alpha=0.1, color="#EF4444")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("Precision-Recall"); ax.legend()
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "04_precision_recall.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  saved {path}")

def plot_feature_importance(model, feature_cols, top_n=15):
    """Top features by importance."""
    imp = model.feature_importances_
    idx = np.argsort(imp)[-top_n:]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(range(top_n), imp[idx], color="#0D9488")
    ax.set_yticks(range(top_n)); ax.set_yticklabels([feature_cols[i] for i in idx])
    ax.set_xlabel("Importance"); ax.set_title("Feature Importance"); plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "05_feature_importance.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  saved {path}")

def plot_timestep(df):
    """Illicit rate per time step."""
    ts = df.groupby("time_step").agg(total=("is_illicit","count"), illicit=("is_illicit","sum"))
    ts["rate"] = ts["illicit"] / ts["total"] * 100
    fig, (ax1,ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    ax1.bar(ts.index, ts["total"], color="#0D9488", alpha=0.7); ax1.set_ylabel("Transactions")
    ax1.set_title("Transactions & Illicit Rate per Time Step")
    ax2.plot(ts.index, ts["rate"], color="#EF4444", linewidth=2, marker="o", markersize=4)
    ax2.fill_between(ts.index, ts["rate"], alpha=0.1, color="#EF4444")
    ax2.set_ylabel("Illicit %"); ax2.set_xlabel("Time Step"); plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "06_timestep.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  saved {path}")


def generate_all():
    """Generate all static charts from the trained model and dataset."""
    from main import load_elliptic, prepare_features
    print("\n  Generating charts...\n")
    DATA_DIR = os.environ.get("DATA_DIR", "data")
    df, _ = load_elliptic(DATA_DIR)
    y = df["is_illicit"]
    plot_class_distribution(y)
    plot_timestep(df)
    if not os.path.exists("models/aml_model.pkl"):
        print("  No model found. Run 'python main.py train' first.")
        return
    model = joblib.load("models/aml_model.pkl")
    feature_cols = joblib.load("models/feature_cols.pkl")
    scaler = joblib.load("models/scaler.pkl")
    pca = joblib.load("models/pca.pkl")
    X, _, _ = prepare_features(df, scaler=scaler, pca=pca, fit=False)
    X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    plot_confusion_matrix(y_test, y_pred)
    plot_roc(y_test, y_proba)
    plot_pr(y_test, y_proba)
    plot_feature_importance(model, feature_cols)
    print(f"\n  All charts saved to {OUTPUT_DIR}/")


# ── Interactive HTML Dashboard ─────────────────────────────

def build_dashboard_html():
    """Build a self-contained interactive HTML dashboard with Plotly.js."""
    from main import load_elliptic, prepare_features, N_RAW_FEATURES

    DATA_DIR = os.environ.get("DATA_DIR", "data")
    df, df_edges = load_elliptic(DATA_DIR)
    y = df["is_illicit"]
    total = len(df); n_ill = int(y.sum()); n_lic = total - n_ill

    ts = df.groupby("time_step").agg(total=("is_illicit","count"), illicit=("is_illicit","sum"))
    ts["rate"] = ts["illicit"] / ts["total"] * 100
    ts_labels = ts.index.tolist()
    ts_counts = ts["total"].tolist()
    ts_rates = [round(r, 2) for r in ts["rate"].tolist()]

    metrics_json = "{}"
    pca_json = "[]"
    fi_labels = "[]"
    fi_values = "[]"
    fi_n95 = 0
    dbscan_json = "[]"
    dbscan_n = 0
    cm_json = "[[0,0],[0,0]]"
    roc_json = '{"fpr":[],"tpr":[],"auc":0}'
    pr_json = '{"rec":[],"prec":[],"auc":0}'

    if os.path.exists("models/aml_model.pkl"):
        model = joblib.load("models/aml_model.pkl")
        feature_cols = joblib.load("models/feature_cols.pkl")
        scaler = joblib.load("models/scaler.pkl")
        pca_model = joblib.load("models/pca.pkl")
        X, _, _ = prepare_features(df, scaler=scaler, pca=pca_model, fit=False)
        X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc_val = roc_auc_score(y_test, y_proba)
        metrics_json = json.dumps({"precision": round(prec,4), "recall": round(rec,4), "f1": round(f1,4), "auc": round(auc_val,4)})
        cm = confusion_matrix(y_test, y_pred).tolist()
        cm_json = json.dumps(cm)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = round(auc(fpr, tpr), 4)
        step = max(1, len(fpr) // 200)
        roc_json = json.dumps({"fpr": fpr[::step].tolist(), "tpr": tpr[::step].tolist(), "auc": roc_auc})
        pr_p, pr_r, _ = precision_recall_curve(y_test, y_proba)
        pr_auc_val = round(auc(pr_r, pr_p), 4)
        step = max(1, len(pr_r) // 200)
        pr_json = json.dumps({"rec": pr_r[::step].tolist(), "prec": pr_p[::step].tolist(), "auc": pr_auc_val})
        raw_cols = [f"feat_{i}" for i in range(N_RAW_FEATURES)]
        sample_n = min(8000, len(df))
        sample_idx = np.random.RandomState(42).choice(len(df), sample_n, replace=False)
        X_sample = scaler.transform(df.iloc[sample_idx][raw_cols].values)
        pca_2d = PCA(n_components=3, random_state=42)
        X_2d = pca_2d.fit_transform(X_sample)
        y_sample = y.iloc[sample_idx].values
        pca_points = [{"x": round(float(X_2d[i,0]),2), "y": round(float(X_2d[i,1]),2), "z": round(float(X_2d[i,2]),2), "c": int(y_sample[i])} for i in range(sample_n)]
        pca_json = json.dumps(pca_points)

        # DBSCAN clustering on illicit transactions
        from sklearn.cluster import DBSCAN
        ill_mask = y_sample == 1
        if ill_mask.sum() > 10:
            X_ill = X_2d[ill_mask]
            db = DBSCAN(eps=1.5, min_samples=5)
            clusters = db.fit_predict(X_ill)
            n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
            dbscan_points = [{"x": round(float(X_ill[i,0]),2), "y": round(float(X_ill[i,1]),2),
                              "z": round(float(X_ill[i,2]),2), "cl": int(clusters[i])} for i in range(len(X_ill))]
            dbscan_json = json.dumps(dbscan_points)
            dbscan_n = n_clusters
        else:
            dbscan_json = "[]"
            dbscan_n = 0
        imp = model.feature_importances_
        top_idx = np.argsort(imp)[-15:][::-1]
        fi_labels = json.dumps([feature_cols[i] for i in top_idx])
        fi_values = json.dumps([round(float(imp[i].item()) / float(imp.sum().item()) * 100, 1) for i in top_idx])
        # Calculate how many features capture 95% of total importance
        sorted_imp = np.sort(imp)[::-1]
        cum_imp = np.cumsum(sorted_imp) / sorted_imp.sum()
        n_95 = int(np.searchsorted(cum_imp, 0.95) + 1)
        fi_n95 = n_95

    sim_rows = "[]"
    sim_path = "data/simulated_months/simulation_summary.json"
    if os.path.exists(sim_path):
        with open(sim_path) as f:
            sim_rows = f.read()

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AML Bitcoin Detection — EDA Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=JetBrains+Mono:wght@400;500&display=swap');
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:'DM Sans',sans-serif;background:#f8fafc;color:#1e293b}}
.topbar{{background:#0F1B2D;padding:1.8rem 3rem;display:flex;justify-content:space-between;align-items:center}}
.topbar h1{{font-size:1.5rem;color:#fff;font-weight:700}}
.topbar .tag{{background:rgba(255,255,255,0.12);color:#94a3b8;padding:0.3rem 0.8rem;border-radius:20px;font-size:0.75rem}}
.container{{max-width:1400px;margin:0 auto;padding:2rem}}
.stats{{display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;margin-bottom:2rem}}
.stat{{background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:1.5rem;text-align:center;transition:box-shadow 0.2s}}
.stat:hover{{box-shadow:0 4px 12px rgba(0,0,0,0.08)}}
.stat .val{{font-size:2.2rem;font-weight:700;font-family:'JetBrains Mono',monospace}}
.stat .label{{color:#64748B;font-size:0.8rem;margin-top:0.3rem;text-transform:uppercase;letter-spacing:1px}}
h2{{color:#0F1B2D;font-size:1.1rem;margin:2.5rem 0 1rem;padding-bottom:0.5rem;border-bottom:2px solid #0D9488;display:flex;align-items:center;gap:0.5rem}}
h2 .dot{{width:8px;height:8px;border-radius:50%;background:#0D9488}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:1.5rem;margin-bottom:1.5rem}}
.card{{background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:1.5rem;min-height:380px}}
.card-wide{{background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:1.5rem;margin-bottom:1.5rem;min-height:400px}}
.card-full{{background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:1.5rem;margin-bottom:1.5rem}}
.card h3{{color:#0F1B2D;font-size:0.85rem;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:0.8rem;font-weight:600}}
.caption{{color:#64748B;font-size:0.8rem;margin-top:0.8rem;line-height:1.6;border-left:3px solid #e2e8f0;padding-left:0.8rem}}
table{{width:100%;border-collapse:separate;border-spacing:0;font-size:0.85rem}}
thead th{{background:#0F1B2D;color:#94a3b8;padding:0.7rem 1rem;text-align:left;font-weight:500;text-transform:uppercase;letter-spacing:1px;font-size:0.7rem}}
thead th:first-child{{border-radius:8px 0 0 0}}thead th:last-child{{border-radius:0 8px 0 0}}
tbody td{{padding:0.6rem 1rem;border-bottom:1px solid #f1f5f9}}tbody tr:hover{{background:#f8fafc}}
.badge{{padding:0.2rem 0.6rem;border-radius:10px;font-size:0.7rem;font-weight:600}}
.badge-red{{background:#fef2f2;color:#dc2626}}.badge-green{{background:#f0fdf4;color:#16a34a}}
.badge-teal{{background:#f0fdfa;color:#0d9488}}.badge-gray{{background:#f8fafc;color:#94a3b8}}
.arch-label{{font-family:'DM Sans',sans-serif;font-weight:500}}
.footer{{text-align:center;padding:3rem;color:#94a3b8;font-size:0.8rem}}
.footer a{{color:#0D9488;text-decoration:none}}
@media(max-width:900px){{.stats{{grid-template-columns:repeat(2,1fr)}}.grid2{{grid-template-columns:1fr}}}}
</style>
</head>
<body>

<div class="topbar">
  <h1>Crypto Fraud Detection</h1>
  <div style="display:flex;gap:0.5rem;align-items:center">
    <span class="tag">Elliptic Dataset</span>
    <span class="tag">XGBoost + PCA</span>
    <span class="tag">MLOps Pipeline</span>
    <button onclick="location.reload()" style="background:#fff;color:#0F1B2D;border:none;padding:0.4rem 1rem;border-radius:20px;font-family:DM Sans;font-size:0.8rem;font-weight:600;cursor:pointer;margin-left:0.5rem;transition:background 0.2s" onmouseover="this.style.background='#f0fdfa'" onmouseout="this.style.background='#fff'">&#x21bb; Refresh data</button>
  </div>
</div>
<div style="background:#0F1B2D;padding:0.5rem 3rem;border-top:1px solid rgba(255,255,255,0.05)">
  <span style="color:#94a3b8;font-size:0.8rem">Objective: Detect illicit cryptocurrency transactions using machine learning, with automated retraining to adapt to evolving criminal tactics.</span>
  <span style="color:#475569;font-size:0.7rem;margin-left:1rem">Last updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</span>
</div>

<div class="container">

<h2><span class="dot"></span> Elliptic Bitcoin Dataset — 203K real blockchain transactions (2017–2018)</h2>
<div class="stats">
  <div class="stat"><div class="val" style="color:#0D9488">{total:,}</div><div class="label">Labeled transactions</div></div>
  <div class="stat"><div class="val" style="color:#16a34a">{n_lic:,}</div><div class="label">Licit (legitimate)</div></div>
  <div class="stat"><div class="val" style="color:#dc2626">{n_ill:,}</div><div class="label">Illicit (fraudulent)</div></div>
  <div class="stat"><div class="val" style="color:#d97706">{n_ill/total*100:.1f}%</div><div class="label">Illicit ratio</div></div>
</div>

<h2><span class="dot"></span> Model performance — XGBoost classifier on 46K labeled transactions</h2>
<div class="grid2">
  <div class="card"><h3>Radar overview</h3><div id="radarChart"></div>
    <p class="caption"><strong>Why these 4 metrics:</strong> Accuracy alone is misleading with 90/10 class imbalance — a model predicting everything as licit would be 90% accurate but useless. Precision measures false alarm rate. Recall measures how many illicit transactions are caught. F1 balances both. AUC measures overall ranking quality regardless of threshold choice.</p>
  </div>
  <div class="card"><h3>Metrics breakdown</h3><div id="metricsBar"></div>
    <p class="caption">Precision = of those flagged, how many were truly illicit. Recall = of all illicit transactions, how many were caught. F1 = balance of both. AUC = overall ranking quality.</p>
  </div>
</div>

<h2><span class="dot"></span> Exploratory data analysis — understanding the fraud landscape</h2>
<div class="grid2">
  <div class="card"><h3>Class distribution</h3><div id="classChart"></div>
    <p class="caption"><strong>The core challenge:</strong> Only ~10% of labeled transactions are illicit. A naive model predicting everything as licit would score 90% accuracy but catch zero fraud. This is why the model uses scale_pos_weight to penalize missed illicit transactions ~9x more than false alarms, and why the evaluation focuses on recall and F1 instead of accuracy.</p>
  </div>
  <div class="card"><h3>Illicit activity across 2 years (2017–2018) — 49 time steps</h3><div id="tsChart"></div>
    <p class="caption"><strong>What this shows:</strong> The Elliptic dataset spans 49 time steps (~2 weeks each, roughly 2 years of Bitcoin activity). The bars show how many transactions occurred per period, the red line shows what percentage were illicit.<br><br><strong>Why it matters:</strong> Criminal activity is not constant — it spikes during specific periods (new laundering schemes, enforcement gaps). This temporal pattern is why the model includes time_step as a feature, and why the MLOps pipeline needs drift detection to catch when these patterns shift.</p>
  </div>
</div>

<h2><span class="dot"></span> Feature space — how the model sees transactions</h2>
<div class="card-wide"><h3>PCA projection — 3D interactive view</h3><div id="pcaChart" style="height:550px"></div>
  <p class="caption"><strong>Why PCA:</strong> The original dataset has 165 anonymized features per transaction — too many to visualize or for the model to process efficiently. PCA compresses them into 30 components that retain ~85% of the information, reducing noise and speeding up training by 5x.<br><br><strong>What this shows:</strong> All 165 features compressed into 3 dimensions. Each dot is one Bitcoin transaction. Drag to rotate. Where red dots cluster separately from teal, the model can distinguish illicit from licit transactions. Where they overlap, classification is harder — those are the borderline cases that drive false positives and false negatives in the confusion matrix above.<br><br><strong>Result:</strong> The visible separation confirms that illicit transactions do have distinct feature patterns (different amounts, timing, network connections) that the model can learn. Without this separation, no classifier would work.</p>
</div>

<div class="card-wide"><h3>DBSCAN clustering — illicit transaction subtypes</h3><div id="dbscanChart" style="height:500px"></div>
  <p class="caption"><strong>Why DBSCAN:</strong> The PCA view shows illicit transactions cluster tightly. DBSCAN (Density-Based Spatial Clustering) identifies distinct groups within the illicit class without requiring a pre-defined number of clusters. Each color represents a different criminal campaign or behavior pattern. Noise points (gray) are outliers that do not fit any pattern — potentially novel laundering techniques.<br><br><strong>Result:</strong> {dbscan_n} distinct clusters found among illicit transactions. This suggests multiple types of criminal activity in the dataset (e.g., ransomware, darknet markets, mixer services). A production system could use these clusters to prioritize investigation by campaign type.</p>
</div>

<div class="card-wide"><h3>Feature importance — top 15</h3><div id="fiChart" style="height:450px"></div>
  <p class="caption"><strong>What this shows:</strong> Which of the 33 model inputs (30 PCA components + 3 graph features) have the most influence on the model's decisions. The top {fi_n95} features capture 95% of total predictive power — the rest contribute marginally.<br><br><strong>In practice:</strong> PCA components are compressed representations of the original 165 blockchain features (transaction amounts, timing patterns, aggregated neighbor stats). Graph features (in_degree = how many transactions send to this one, out_degree = how many it sends to) capture network structure. High importance on graph features means the model is learning that connected transaction patterns matter — which is exactly how money laundering works (layering through chains of transactions).</p>
</div>

<h2><span class="dot"></span> Model evaluation — where the classifier gets it right and wrong</h2>
<div class="card-wide"><h3>Confusion matrix</h3>
  <div id="cmChart" style="height:400px"></div>
  <p class="caption"><strong>How to read:</strong> Green = correct, red = errors. TN (top-left): licit transactions correctly cleared — no action needed. TP (bottom-right): illicit transactions correctly caught — flagged for investigation. FP (bottom-left): licit transactions falsely flagged — wastes analyst time but causes no harm. FN (top-right): illicit transactions missed — the most dangerous error, money laundering goes undetected.<br><br><strong>In AML practice:</strong> Regulators care most about minimizing FN (missed illicit). Banks accept some FP (false alarms) as a cost of compliance. The model's recall score directly reflects the FN rate.</p>
</div>

<h2><span class="dot"></span> REST API — 7 endpoints serving the model</h2>
<div class="card-full" style="padding:2rem">
  <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;margin-bottom:1rem">
    <div style="background:#f0fdf4;border:1px solid #16a34a;border-radius:10px;padding:1rem;text-align:center">
      <div style="font-size:0.7rem;color:#16a34a;font-weight:600;text-transform:uppercase;letter-spacing:1px">GET</div>
      <div style="font-family:JetBrains Mono,monospace;font-size:1.1rem;color:#0F1B2D;font-weight:600;margin:0.4rem 0">/health</div>
      <div style="font-size:0.75rem;color:#64748B">API status check<br>No auth required</div>
    </div>
    <div style="background:#eff6ff;border:1px solid #2563eb;border-radius:10px;padding:1rem;text-align:center">
      <div style="font-size:0.7rem;color:#2563eb;font-weight:600;text-transform:uppercase;letter-spacing:1px">GET</div>
      <div style="font-family:JetBrains Mono,monospace;font-size:1.1rem;color:#0F1B2D;font-weight:600;margin:0.4rem 0">/dashboard</div>
      <div style="font-size:0.75rem;color:#64748B">Interactive EDA<br>Live refresh</div>
    </div>
    <div style="background:#f0fdfa;border:1px solid #0d9488;border-radius:10px;padding:1rem;text-align:center">
      <div style="font-size:0.7rem;color:#0d9488;font-weight:600;text-transform:uppercase;letter-spacing:1px">POST</div>
      <div style="font-family:JetBrains Mono,monospace;font-size:1.1rem;color:#0F1B2D;font-weight:600;margin:0.4rem 0">/predict</div>
      <div style="font-size:0.75rem;color:#64748B">Score transaction<br>Returns probability + risk</div>
    </div>
    <div style="background:#f0fdfa;border:1px solid #0d9488;border-radius:10px;padding:1rem;text-align:center">
      <div style="font-size:0.7rem;color:#0d9488;font-weight:600;text-transform:uppercase;letter-spacing:1px">POST</div>
      <div style="font-family:JetBrains Mono,monospace;font-size:1.1rem;color:#0F1B2D;font-weight:600;margin:0.4rem 0">/predict/batch</div>
      <div style="font-size:0.75rem;color:#64748B">Score multiple<br>Bulk processing</div>
    </div>
  </div>
  <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:1rem">
    <div style="background:#fffbeb;border:1px solid #d97706;border-radius:10px;padding:1rem;text-align:center">
      <div style="font-size:0.7rem;color:#d97706;font-weight:600;text-transform:uppercase;letter-spacing:1px">GET</div>
      <div style="font-family:JetBrains Mono,monospace;font-size:1.1rem;color:#0F1B2D;font-weight:600;margin:0.4rem 0">/model/info</div>
      <div style="font-size:0.75rem;color:#64748B">Model metadata, version<br>Training timestamp</div>
    </div>
    <div style="background:#fef2f2;border:1px solid #dc2626;border-radius:10px;padding:1rem;text-align:center">
      <div style="font-size:0.7rem;color:#dc2626;font-weight:600;text-transform:uppercase;letter-spacing:1px">GET</div>
      <div style="font-family:JetBrains Mono,monospace;font-size:1.1rem;color:#0F1B2D;font-weight:600;margin:0.4rem 0">/alerts</div>
      <div style="font-size:0.75rem;color:#64748B">Drift alert history<br>Feature-level detail</div>
    </div>
    <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:1rem;text-align:center">
      <div style="font-size:0.7rem;color:#64748B;font-weight:600;text-transform:uppercase;letter-spacing:1px">ALL</div>
      <div style="font-family:JetBrains Mono,monospace;font-size:1.1rem;color:#0F1B2D;font-weight:600;margin:0.4rem 0">X-API-Key</div>
      <div style="font-size:0.75rem;color:#64748B">Auth required on<br>/predict, /model, /alerts</div>
    </div>
  </div>
  <p class="caption" style="margin-top:1rem"><strong>Access control:</strong> Prediction and monitoring endpoints require an API key via the X-API-Key header. Unauthorized requests are logged with source IP and rejected with HTTP 401. The /health and /dashboard endpoints are public for load balancer checks and stakeholder access.</p>
</div>

<h2><span class="dot"></span> System architecture — from raw data to production deployment</h2>
<div class="card-full" style="padding:2rem;overflow-x:auto">
<svg width="100%" viewBox="0 0 920 580" style="max-width:920px;margin:0 auto;display:block">
  <defs>
    <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M1 1L9 5L1 9" fill="none" stroke="context-stroke" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
    </marker>
    <filter id="sh"><feDropShadow dx="0" dy="2" stdDeviation="3" flood-opacity="0.06"/></filter>
  </defs>

  <!-- ── DATA LAYER ── -->
  <text x="30" y="18" style="font-family:DM Sans;font-size:11px;fill:#94a3b8;letter-spacing:3px;font-weight:600">DATA LAYER</text>
  <rect x="30" y="30" width="180" height="80" rx="10" fill="#fff" stroke="#0d9488" stroke-width="1.5" filter="url(#sh)"/>
  <text x="120" y="58" text-anchor="middle" style="font-family:DM Sans;font-size:14px;fill:#0d9488;font-weight:700">Elliptic dataset</text>
  <text x="120" y="76" text-anchor="middle" style="font-family:DM Sans;font-size:10px;fill:#64748B">3 CSVs · 203K transactions</text>
  <text x="120" y="92" text-anchor="middle" style="font-family:DM Sans;font-size:10px;fill:#64748B">165 features · 234K edges</text>

  <!-- relationship: feeds into -->
  <line x1="210" y1="70" x2="268" y2="70" stroke="#0d9488" stroke-width="1.5" marker-end="url(#arr)"/>
  <text x="239" y="62" text-anchor="middle" style="font-family:DM Sans;font-size:9px;fill:#0d9488;font-style:italic">feeds</text>

  <!-- ── PROCESSING LAYER ── -->
  <text x="270" y="18" style="font-family:DM Sans;font-size:11px;fill:#94a3b8;letter-spacing:3px;font-weight:600">PROCESSING</text>
  <rect x="272" y="30" width="170" height="80" rx="10" fill="#fff" stroke="#0d9488" stroke-width="1.5" filter="url(#sh)"/>
  <text x="357" y="58" text-anchor="middle" style="font-family:DM Sans;font-size:14px;fill:#0d9488;font-weight:700">Feature pipeline</text>
  <text x="357" y="76" text-anchor="middle" style="font-family:DM Sans;font-size:10px;fill:#64748B">StandardScaler</text>
  <text x="357" y="92" text-anchor="middle" style="font-family:DM Sans;font-size:10px;fill:#64748B">PCA: 165 → 30 components</text>

  <!-- relationship: transforms to -->
  <line x1="442" y1="70" x2="498" y2="70" stroke="#7c3aed" stroke-width="1.5" marker-end="url(#arr)"/>
  <text x="470" y="62" text-anchor="middle" style="font-family:DM Sans;font-size:9px;fill:#7c3aed;font-style:italic">vectors</text>

  <!-- ── MODEL LAYER ── -->
  <text x="500" y="18" style="font-family:DM Sans;font-size:11px;fill:#94a3b8;letter-spacing:3px;font-weight:600">MODEL</text>
  <rect x="502" y="30" width="160" height="80" rx="10" fill="#fff" stroke="#7c3aed" stroke-width="1.5" filter="url(#sh)"/>
  <text x="582" y="58" text-anchor="middle" style="font-family:DM Sans;font-size:14px;fill:#7c3aed;font-weight:700">XGBoost</text>
  <text x="582" y="76" text-anchor="middle" style="font-family:DM Sans;font-size:10px;fill:#64748B">200 trees · depth 6</text>
  <text x="582" y="92" text-anchor="middle" style="font-family:DM Sans;font-size:10px;fill:#64748B">scale_pos_weight · AUCPR</text>

  <!-- relationship: scores -->
  <line x1="662" y1="70" x2="718" y2="70" stroke="#2563eb" stroke-width="1.5" marker-end="url(#arr)"/>
  <text x="690" y="62" text-anchor="middle" style="font-family:DM Sans;font-size:9px;fill:#2563eb;font-style:italic">scores</text>

  <!-- ── SERVING LAYER ── -->
  <text x="720" y="18" style="font-family:DM Sans;font-size:11px;fill:#94a3b8;letter-spacing:3px;font-weight:600">SERVING</text>
  <rect x="722" y="30" width="170" height="80" rx="10" fill="#fff" stroke="#2563eb" stroke-width="1.5" filter="url(#sh)"/>
  <text x="807" y="52" text-anchor="middle" style="font-family:DM Sans;font-size:14px;fill:#2563eb;font-weight:700">Flask REST API</text>
  <text x="807" y="70" text-anchor="middle" style="font-family:DM Sans;font-size:10px;fill:#64748B">/predict · /batch · /health</text>
  <text x="807" y="86" text-anchor="middle" style="font-family:DM Sans;font-size:10px;fill:#64748B">API key auth · Docker</text>
  <text x="807" y="100" text-anchor="middle" style="font-family:DM Sans;font-size:10px;fill:#7c3aed;font-weight:500">→ Render.com cloud</text>

  <!-- OUTPUT: risk levels -->
  <line x1="807" y1="110" x2="807" y2="148" stroke="#2563eb" stroke-width="1" stroke-dasharray="4 3" marker-end="url(#arr)"/>
  <text x="830" y="136" style="font-family:DM Sans;font-size:9px;fill:#64748B;font-style:italic">returns</text>
  <rect x="722" y="152" width="50" height="28" rx="5" fill="#f0fdf4" stroke="#16a34a" stroke-width="1"/>
  <text x="747" y="170" text-anchor="middle" style="font-family:DM Sans;font-size:11px;fill:#16a34a;font-weight:600">LOW</text>
  <rect x="780" y="152" width="50" height="28" rx="5" fill="#fffbeb" stroke="#d97706" stroke-width="1"/>
  <text x="805" y="170" text-anchor="middle" style="font-family:DM Sans;font-size:11px;fill:#d97706;font-weight:600">MED</text>
  <rect x="838" y="152" width="54" height="28" rx="5" fill="#fef2f2" stroke="#dc2626" stroke-width="1"/>
  <text x="865" y="170" text-anchor="middle" style="font-family:DM Sans;font-size:11px;fill:#dc2626;font-weight:600">HIGH</text>

  <!-- ── SEPARATOR ── -->
  <line x1="30" y1="210" x2="892" y2="210" stroke="#e2e8f0" stroke-width="1.5"/>
  <text x="460" y="236" text-anchor="middle" style="font-family:DM Sans;font-size:11px;fill:#94a3b8;letter-spacing:3px;font-weight:600">MLOPS LAYER</text>

  <!-- ── MONITORING ── -->
  <rect x="30" y="252" width="200" height="80" rx="10" fill="#fff" stroke="#d97706" stroke-width="1.5" filter="url(#sh)"/>
  <text x="130" y="280" text-anchor="middle" style="font-family:DM Sans;font-size:14px;fill:#d97706;font-weight:700">MLflow</text>
  <text x="130" y="298" text-anchor="middle" style="font-family:DM Sans;font-size:10px;fill:#64748B">Experiment tracking</text>
  <text x="130" y="314" text-anchor="middle" style="font-family:DM Sans;font-size:10px;fill:#64748B">Model versioning · Metrics</text>

  <!-- relationship: logs to -->
  <line x1="582" y1="110" x2="582" y2="190" stroke="#d97706" stroke-width="1" stroke-dasharray="4 3"/>
  <line x1="582" y1="190" x2="130" y2="190" stroke="#d97706" stroke-width="1" stroke-dasharray="4 3"/>
  <line x1="130" y1="190" x2="130" y2="248" stroke="#d97706" stroke-width="1" stroke-dasharray="4 3" marker-end="url(#arr)"/>
  <text x="356" y="186" text-anchor="middle" style="font-family:DM Sans;font-size:9px;fill:#d97706;font-style:italic">logs params, metrics, artifacts</text>

  <!-- relationship: monitors -->
  <line x1="230" y1="292" x2="328" y2="292" stroke="#ea580c" stroke-width="1.5" marker-end="url(#arr)"/>
  <text x="279" y="284" text-anchor="middle" style="font-family:DM Sans;font-size:9px;fill:#ea580c;font-style:italic">monitors</text>

  <!-- ── DRIFT DETECTION ── -->
  <rect x="332" y="252" width="210" height="80" rx="10" fill="#fff" stroke="#ea580c" stroke-width="1.5" filter="url(#sh)"/>
  <text x="437" y="280" text-anchor="middle" style="font-family:DM Sans;font-size:14px;fill:#ea580c;font-weight:700">Drift detection</text>
  <text x="437" y="298" text-anchor="middle" style="font-family:DM Sans;font-size:10px;fill:#64748B">Kolmogorov-Smirnov test</text>
  <text x="437" y="314" text-anchor="middle" style="font-family:DM Sans;font-size:10px;fill:#64748B">p &lt; 0.05 · 30% feature threshold</text>

  <!-- relationship: triggers -->
  <line x1="542" y1="292" x2="638" y2="292" stroke="#16a34a" stroke-width="1.5" marker-end="url(#arr)"/>
  <text x="590" y="284" text-anchor="middle" style="font-family:DM Sans;font-size:9px;fill:#16a34a;font-style:italic">triggers</text>

  <!-- ── CI/CD ── -->
  <rect x="642" y="252" width="250" height="80" rx="10" fill="#fff" stroke="#16a34a" stroke-width="1.5" filter="url(#sh)"/>
  <text x="767" y="280" text-anchor="middle" style="font-family:DM Sans;font-size:14px;fill:#16a34a;font-weight:700">GitHub Actions CI/CD</text>
  <text x="767" y="298" text-anchor="middle" style="font-family:DM Sans;font-size:10px;fill:#64748B">Monthly cron · Drift trigger · Manual</text>
  <text x="767" y="314" text-anchor="middle" style="font-family:DM Sans;font-size:10px;fill:#64748B">Retrain → Validate → Deploy</text>

  <!-- RETRAIN FEEDBACK LOOP -->
  <path d="M767 332 L767 370 L582 370 L582 110" fill="none" stroke="#16a34a" stroke-width="2" stroke-dasharray="6 3" marker-end="url(#arr)"/>
  <rect x="620" y="355" width="110" height="22" rx="4" fill="#f0fdf4" stroke="#16a34a" stroke-width="0.5"/>
  <text x="675" y="370" text-anchor="middle" style="font-family:DM Sans;font-size:10px;fill:#16a34a;font-weight:600">retrains model</text>

  <!-- ── SEPARATOR ── -->
  <line x1="30" y1="400" x2="892" y2="400" stroke="#e2e8f0" stroke-width="1.5"/>
  <text x="460" y="426" text-anchor="middle" style="font-family:DM Sans;font-size:11px;fill:#94a3b8;letter-spacing:3px;font-weight:600">CI/CD PIPELINE</text>

  <!-- Pipeline steps with arrows -->
  <rect x="40" y="440" width="145" height="45" rx="8" fill="#f0fdfa" stroke="#0d9488" stroke-width="1"/>
  <text x="112" y="467" text-anchor="middle" style="font-family:DM Sans;font-size:12px;fill:#0d9488;font-weight:600">1 · Drift check</text>
  <line x1="185" y1="462" x2="213" y2="462" stroke="#0d9488" stroke-width="1" marker-end="url(#arr)"/>

  <rect x="217" y="440" width="130" height="45" rx="8" fill="#f0fdfa" stroke="#0d9488" stroke-width="1"/>
  <text x="282" y="467" text-anchor="middle" style="font-family:DM Sans;font-size:12px;fill:#0d9488;font-weight:600">2 · Retrain</text>
  <line x1="347" y1="462" x2="375" y2="462" stroke="#0d9488" stroke-width="1" marker-end="url(#arr)"/>

  <rect x="379" y="440" width="130" height="45" rx="8" fill="#f0fdfa" stroke="#0d9488" stroke-width="1"/>
  <text x="444" y="467" text-anchor="middle" style="font-family:DM Sans;font-size:12px;fill:#0d9488;font-weight:600">3 · Validate</text>
  <line x1="509" y1="462" x2="537" y2="462" stroke="#0d9488" stroke-width="1" marker-end="url(#arr)"/>

  <rect x="541" y="440" width="155" height="45" rx="8" fill="#f0fdfa" stroke="#0d9488" stroke-width="1"/>
  <text x="618" y="467" text-anchor="middle" style="font-family:DM Sans;font-size:12px;fill:#0d9488;font-weight:600">4 · Docker build</text>
  <line x1="696" y1="462" x2="724" y2="462" stroke="#0d9488" stroke-width="1" marker-end="url(#arr)"/>

  <rect x="728" y="440" width="160" height="45" rx="8" fill="#f0fdfa" stroke="#0d9488" stroke-width="1"/>
  <text x="808" y="467" text-anchor="middle" style="font-family:DM Sans;font-size:12px;fill:#0d9488;font-weight:600">5 · Health check</text>

  <!-- Footer -->
  <text x="460" y="520" text-anchor="middle" style="font-family:DM Sans;font-size:10px;fill:#94a3b8">Iman Jouhar · DLBDSMTP01 · IU International University of Applied Sciences</text>
  <text x="460" y="540" text-anchor="middle" style="font-family:DM Sans;font-size:9px;fill:#cbd5e1">Data flows left→right through the prediction pipeline. The MLOps layer monitors for drift and triggers automated retraining via GitHub Actions.</text>
</svg>
</div>

<h2><span class="dot"></span> 12-month drift simulation — testing the automated retraining pipeline</h2>
<div class="card-wide"><h3>Monthly fraud rate — drift phases and retraining events</h3><div id="simChart" style="height:450px"></div>
  <p class="caption"><strong>What this tests:</strong> Criminals evolve their tactics over time. A frozen model degrades as data shifts away from its training distribution. This simulation applies progressive noise across 12 monthly windows to validate that the MLOps pipeline detects the change and retrains before performance drops.<br><br><strong>How to read:</strong> Each bar is one month's fraud rate. Color indicates the drift phase: green = stable (no drift), yellow = mild drift, orange = moderate drift, red = strong drift. Green stars mark months where the model was automatically retrained. The pipeline should retrain at or before the transition from stable to drifted data — confirming the KS test threshold works correctly.</p>
</div>

<div class="footer">
  Crypto Fraud Detection &middot; Iman Jouhar &middot; DLBDSMTP01<br>
  <a href="https://github.com/imanjouhar/crypto-fraud-detection">GitHub Repository</a>
</div>

</div>

<script>
const L = {{paper_bgcolor:'transparent',plot_bgcolor:'transparent',
  font:{{family:'DM Sans',color:'#475569',size:13}},
  margin:{{t:40,r:50,b:60,l:70}},
  xaxis:{{gridcolor:'#f1f5f9',zerolinecolor:'#e2e8f0',tickfont:{{size:11}}}},
  yaxis:{{gridcolor:'#f1f5f9',zerolinecolor:'#e2e8f0',tickfont:{{size:11}}}}
}};
const cfg = {{displayModeBar:false,responsive:true}};

const m = {metrics_json};
if(m.precision) {{
  Plotly.newPlot('radarChart',[{{
    type:'scatterpolar',r:[m.precision,m.recall,m.f1,m.auc,m.precision],
    theta:['Precision','Recall','F1 Score','ROC-AUC','Precision'],
    fill:'toself',fillcolor:'rgba(13,148,136,0.12)',
    line:{{color:'#0D9488',width:2.5}},marker:{{size:7,color:'#0D9488'}}
  }}],{{...L,polar:{{bgcolor:'transparent',
    radialaxis:{{range:[0,1],gridcolor:'#f1f5f9',linecolor:'#e2e8f0',tickfont:{{color:'#94a3b8',size:10}}}},
    angularaxis:{{gridcolor:'#f1f5f9',linecolor:'#e2e8f0',tickfont:{{color:'#1e293b',size:13}}}}
  }}}},cfg);
  Plotly.newPlot('metricsBar',[{{
    x:['Precision','Recall','F1 Score','ROC-AUC'],y:[m.precision,m.recall,m.f1,m.auc],type:'bar',
    marker:{{color:['#0D9488','#14B8A6','#d97706','#6366f1'],opacity:0.85}},
    text:[m.precision,m.recall,m.f1,m.auc].map(v=>v.toFixed(2)),textposition:'outside',textfont:{{color:'#1e293b',size:15}}
  }}],{{...L,yaxis:{{...L.yaxis,range:[0,1.12],title:'Score'}}}},cfg);
}}

Plotly.newPlot('classChart',[{{
  x:['Licit','Illicit'],y:[{n_lic},{n_ill}],type:'bar',
  marker:{{color:['#0D9488','#dc2626'],opacity:0.85}},
  text:['{n_lic:,}','{n_ill:,}'],textposition:'outside',textfont:{{color:'#1e293b'}}
}}],{{...L,yaxis:{{...L.yaxis,title:'Count'}}}},cfg);

Plotly.newPlot('tsChart',[
  {{x:{json.dumps(ts_labels)},y:{json.dumps(ts_counts)},type:'bar',name:'Volume',
    marker:{{color:'#0D9488',opacity:0.3}},yaxis:'y'}},
  {{x:{json.dumps(ts_labels)},y:{json.dumps(ts_rates)},type:'scatter',mode:'lines+markers',
    name:'Illicit rate (%)',line:{{color:'#dc2626',width:2.5}},marker:{{size:5,color:'#dc2626'}},yaxis:'y2'}}
],{{...L,
  xaxis:{{...L.xaxis,title:'Time step (~2 weeks each)',dtick:5}},
  yaxis:{{...L.yaxis,title:'Transaction volume',side:'left'}},
  yaxis2:{{title:'Illicit rate (%)',overlaying:'y',side:'right',gridcolor:'transparent',color:'#dc2626',rangemode:'tozero'}},
  legend:{{x:0,y:1.12,orientation:'h',font:{{size:11}}}}
}},cfg);

const pcaData = {pca_json};
if(pcaData.length > 0) {{
  const lic = pcaData.filter(p=>p.c===0);
  const ill = pcaData.filter(p=>p.c===1);
  Plotly.newPlot('pcaChart',[
    {{x:lic.map(p=>p.x),y:lic.map(p=>p.y),z:lic.map(p=>p.z),mode:'markers',name:'Licit ('+lic.length.toLocaleString()+' · ~90%)',
      marker:{{color:'#0D9488',size:3,opacity:0.2}},type:'scatter3d'}},
    {{x:ill.map(p=>p.x),y:ill.map(p=>p.y),z:ill.map(p=>p.z),mode:'markers',name:'Illicit ('+ill.length.toLocaleString()+' · ~10%)',
      marker:{{color:'#dc2626',size:3,opacity:0.5}},type:'scatter3d'}}
  ],{{...L,height:550,margin:{{t:0,r:0,b:0,l:0}},
    scene:{{xaxis:{{title:'PC1',gridcolor:'#f1f5f9',backgroundcolor:'#fafafa'}},
            yaxis:{{title:'PC2',gridcolor:'#f1f5f9',backgroundcolor:'#fafafa'}},
            zaxis:{{title:'PC3',gridcolor:'#f1f5f9',backgroundcolor:'#fafafa'}},
            bgcolor:'#fafafa'}},
    legend:{{x:0,y:1,font:{{size:12}}}}}},cfg);
}}

// DBSCAN clustering
const dbData = {dbscan_json};
if(dbData.length > 0) {{
  const clusterIds = [...new Set(dbData.map(p=>p.cl))].sort((a,b)=>a-b);
  const palette = ['#dc2626','#2563eb','#16a34a','#d97706','#7c3aed','#ec4899','#06b6d4','#84cc16'];
  const traces = clusterIds.map((cl,i) => {{
    const pts = dbData.filter(p=>p.cl===cl);
    const name = cl === -1 ? 'Noise (outliers)' : 'Cluster '+cl+' ('+pts.length+' txns)';
    const color = cl === -1 ? '#94a3b8' : palette[i % palette.length];
    return {{
      x:pts.map(p=>p.x),y:pts.map(p=>p.y),z:pts.map(p=>p.z),
      mode:'markers',name:name,type:'scatter3d',
      marker:{{size:cl===-1?2:4,color:color,opacity:cl===-1?0.3:0.7}}
    }};
  }});
  Plotly.newPlot('dbscanChart',traces,{{
    paper_bgcolor:'transparent',plot_bgcolor:'transparent',
    font:{{family:'DM Sans',color:'#475569',size:12}},
    height:500,margin:{{t:0,r:0,b:0,l:0}},
    scene:{{xaxis:{{title:'PC1',gridcolor:'#f1f5f9',backgroundcolor:'#fafafa'}},
            yaxis:{{title:'PC2',gridcolor:'#f1f5f9',backgroundcolor:'#fafafa'}},
            zaxis:{{title:'PC3',gridcolor:'#f1f5f9',backgroundcolor:'#fafafa'}},
            bgcolor:'#fafafa'}},
    legend:{{x:0,y:1,font:{{size:11}}}}
  }},cfg);
}}

const fiL = {fi_labels};
const fiV = {fi_values};
if(fiL.length > 0) {{
  Plotly.newPlot('fiChart',[{{
    y:fiL.slice().reverse(),x:fiV.slice().reverse(),type:'bar',orientation:'h',
    marker:{{color:fiV.slice().reverse().map((v,i)=>i<3?'#0D9488':'#b2dfdb')}},
    text:fiV.slice().reverse().map(v=>v.toFixed(1)+'%'),textposition:'outside',textfont:{{color:'#475569',size:11}}
  }}],{{...L,height:450,margin:{{t:20,r:80,b:40,l:120}},xaxis:{{...L.xaxis,title:'Importance (%)',ticksuffix:'%'}}}},cfg);
}}

const cm = {cm_json};
const cmColors = [['#dcfce7','#fecaca'],['#fecaca','#dcfce7']];
Plotly.newPlot('cmChart',[{{
  z:[[1,0],[0,1]],x:['Predicted: Licit','Predicted: Illicit'],y:['Actual: Licit','Actual: Illicit'],type:'heatmap',
  colorscale:[[0,'#fecaca'],[1,'#dcfce7']],showscale:false,
  text:[[cm[0][0].toLocaleString()+'\\nTrue Negative (TN)',cm[0][1].toLocaleString()+'\\nFalse Positive (FP)'],[cm[1][0].toLocaleString()+'\\nFalse Negative (FN)',cm[1][1].toLocaleString()+'\\nTrue Positive (TP)']],
  texttemplate:'%{{text}}',textfont:{{size:15,color:'#1e293b'}}
}}],{{...L,height:400,margin:{{t:20,r:30,b:50,l:120}},xaxis:{{...L.xaxis}},yaxis:{{...L.yaxis,autorange:'reversed'}}}},cfg);

const simData = {sim_rows};
if(simData.length > 0) {{
  // Build individual bars with correct heights
  const barData = simData.map(s => {{
    let color = '#16a34a';
    let phase = 'Stable';
    if(s.month > 9) {{ color = '#dc2626'; phase = 'Strong drift'; }}
    else if(s.month > 6) {{ color = '#ea580c'; phase = 'Moderate drift'; }}
    else if(s.month > 3) {{ color = '#eab308'; phase = 'Mild drift'; }}
    const label = s.fraud_rate_pct + '%' + (s.retrained ? ' ★' : '');
    return {{ month: s.month, rate: s.fraud_rate_pct, color: color, label: label, phase: phase, retrained: s.retrained }};
  }});

  Plotly.newPlot('simChart',[{{
    x: barData.map(b => 'Month ' + b.month),
    y: barData.map(b => b.rate),
    type: 'bar',
    marker: {{ color: barData.map(b => b.color), opacity: 0.85 }},
    text: barData.map(b => b.label),
    textposition: 'outside',
    textfont: {{ size: 12, color: '#1e293b' }},
    hovertemplate: barData.map(b => b.phase + '<br>Fraud: ' + b.rate + '%' + (b.retrained ? '<br>Model retrained' : '') + '<extra></extra>')
  }}],{{
    paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
    font: {{ family: 'DM Sans', color: '#475569', size: 13 }},
    height: 450,
    margin: {{ t: 50, r: 30, b: 70, l: 60 }},
    xaxis: {{ gridcolor: '#f1f5f9', tickangle: -45, tickfont: {{ size: 11 }} }},
    yaxis: {{ gridcolor: '#f1f5f9', title: 'Fraud rate (%)', rangemode: 'tozero', ticksuffix: '%', dtick: 5 }},
    showlegend: false,
    annotations: [
      {{ x: 0.12, y: 1.06, xref: 'paper', yref: 'paper', text: '<b>Stable</b>', font: {{ size: 11, color: '#16a34a' }}, showarrow: false }},
      {{ x: 0.37, y: 1.06, xref: 'paper', yref: 'paper', text: '<b>Mild drift</b>', font: {{ size: 11, color: '#eab308' }}, showarrow: false }},
      {{ x: 0.62, y: 1.06, xref: 'paper', yref: 'paper', text: '<b>Moderate</b>', font: {{ size: 11, color: '#ea580c' }}, showarrow: false }},
      {{ x: 0.87, y: 1.06, xref: 'paper', yref: 'paper', text: '<b>Strong drift</b>', font: {{ size: 11, color: '#dc2626' }}, showarrow: false }}
    ],
    shapes: [
      {{ type: 'line', x0: 0.25, x1: 0.25, y0: 0, y1: 1, xref: 'paper', yref: 'paper', line: {{ color: '#e2e8f0', width: 1, dash: 'dot' }} }},
      {{ type: 'line', x0: 0.5, x1: 0.5, y0: 0, y1: 1, xref: 'paper', yref: 'paper', line: {{ color: '#e2e8f0', width: 1, dash: 'dot' }} }},
      {{ type: 'line', x0: 0.75, x1: 0.75, y0: 0, y1: 1, xref: 'paper', yref: 'paper', line: {{ color: '#e2e8f0', width: 1, dash: 'dot' }} }}
    ]
  }},cfg);
}} else {{
  document.getElementById('simChart').innerHTML = '<p style="padding:2rem;color:#94a3b8;text-align:center">Run python main.py simulate first</p>';
}}
</script>

</body>
</html>"""
    return html


def launch_dashboard():
    """Generate the interactive HTML dashboard and open it in the browser."""
    print("  Building interactive dashboard...")
    html = build_dashboard_html()
    path = os.path.join(OUTPUT_DIR, "dashboard.html")
    with open(path, "w") as f:
        f.write(html)
    print(f"  Dashboard saved: {path}")
    import webbrowser
    webbrowser.open(f"file://{os.path.abspath(path)}")
    return path


if __name__ == "__main__":
    generate_all()
    launch_dashboard()