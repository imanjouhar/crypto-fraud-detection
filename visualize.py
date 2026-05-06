"""
Visualizations – AML Bitcoin Fraud Detection (Elliptic Dataset)
================================================================
Generates charts: PCA scatter, UMAP projection, confusion matrix,
ROC, precision-recall, feature importance, drift timeline, dashboard.

Usage:
    python main.py visualize
    python main.py dashboard
"""

import os, json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    precision_recall_curve, classification_report
)
from sklearn.model_selection import train_test_split
import joblib

COLORS = {
    "teal": "#0D9488", "navy": "#0F1B2D", "mint": "#14B8A6",
    "red": "#EF4444", "amber": "#F59E0B", "green": "#22C55E",
    "gray": "#64748B", "light": "#F0F4F8", "purple": "#8B5CF6",
}

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.3, "font.family": "sans-serif",
    "font.size": 11, "axes.titlesize": 14, "axes.titleweight": "bold",
})

OUTPUT_DIR = "visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. CLASS DISTRIBUTION
# ---------------------------------------------------------------------------

def plot_class_distribution(y):
    counts = y.value_counts().sort_index()
    labels = ["Licit", "Illicit"]
    colors = [COLORS["teal"], COLORS["red"]]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, counts.values, color=colors, width=0.5)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                f"{val:,}\n({val/len(y)*100:.1f}%)", ha="center", fontsize=12, fontweight="bold")
    ax.set_ylabel("Transactions")
    ax.set_title("Class Distribution – Elliptic Bitcoin Dataset")
    ax.set_ylim(0, counts.max() * 1.15)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "01_class_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  ✓ {path}"); return path


# ---------------------------------------------------------------------------
# 2. PCA SCATTER (2D projection)
# ---------------------------------------------------------------------------

def plot_pca_scatter(X, y, scaler, pca_model, df):
    """2D PCA projection of all labeled transactions."""
    from sklearn.decomposition import PCA

    raw_cols = [f"feat_{i}" for i in range(165)]
    X_scaled = scaler.transform(df[raw_cols].values)
    pca_2d = PCA(n_components=2, random_state=42)
    X_2d = pca_2d.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(10, 7))
    licit_mask = y == 0
    ax.scatter(X_2d[licit_mask, 0], X_2d[licit_mask, 1], c=COLORS["teal"],
               alpha=0.15, s=5, label="Licit", rasterized=True)
    ax.scatter(X_2d[~licit_mask, 0], X_2d[~licit_mask, 1], c=COLORS["red"],
               alpha=0.6, s=15, label="Illicit", rasterized=True)

    explained = pca_2d.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({explained[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({explained[1]*100:.1f}%)")
    ax.set_title("PCA Projection – Licit vs Illicit Transactions")
    ax.legend(markerscale=3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "02_pca_scatter.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  ✓ {path}"); return path


# ---------------------------------------------------------------------------
# 3. UMAP PROJECTION
# ---------------------------------------------------------------------------

def plot_umap_scatter(df, y):
    """UMAP 2D embedding of transactions (subsample for speed)."""
    try:
        from umap import UMAP
    except ImportError:
        print("  ⚠ umap-learn not installed, skipping UMAP plot")
        return None

    # Subsample for speed
    n_sample = min(15000, len(df))
    idx = np.random.RandomState(42).choice(len(df), n_sample, replace=False)

    raw_cols = [f"feat_{i}" for i in range(165)]
    X_sub = df.iloc[idx][raw_cols].values
    y_sub = y.iloc[idx].values

    print("  Computing UMAP (this may take a minute) …")
    reducer = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    X_umap = reducer.fit_transform(X_sub)

    fig, ax = plt.subplots(figsize=(10, 7))
    licit = y_sub == 0
    ax.scatter(X_umap[licit, 0], X_umap[licit, 1], c=COLORS["teal"],
               alpha=0.2, s=5, label="Licit", rasterized=True)
    ax.scatter(X_umap[~licit, 0], X_umap[~licit, 1], c=COLORS["red"],
               alpha=0.7, s=15, label="Illicit", rasterized=True)

    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.set_title(f"UMAP Projection – {n_sample:,} Transactions")
    ax.legend(markerscale=3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "03_umap_scatter.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  ✓ {path}"); return path


# ---------------------------------------------------------------------------
# 4. CONFUSION MATRIX
# ---------------------------------------------------------------------------

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    labels = ["Licit", "Illicit"]
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max()/2 else COLORS["navy"]
            ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center", fontsize=18, fontweight="bold", color=color)
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix"); plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "04_confusion_matrix.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  ✓ {path}"); return path


# ---------------------------------------------------------------------------
# 5. ROC CURVE
# ---------------------------------------------------------------------------

def plot_roc_curve(y_true, y_proba):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color=COLORS["teal"], linewidth=2.5, label=f"XGBoost (AUC = {roc_auc:.4f})")
    ax.plot([0,1], [0,1], "k--", alpha=0.4, label="Random")
    ax.fill_between(fpr, tpr, alpha=0.1, color=COLORS["teal"])
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve – Bitcoin AML Detection"); ax.legend()
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "05_roc_curve.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  ✓ {path}"); return path


# ---------------------------------------------------------------------------
# 6. PRECISION-RECALL
# ---------------------------------------------------------------------------

def plot_precision_recall(y_true, y_proba):
    prec, rec, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(rec, prec)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(rec, prec, color=COLORS["red"], linewidth=2.5, label=f"PR-AUC = {pr_auc:.4f}")
    ax.fill_between(rec, prec, alpha=0.1, color=COLORS["red"])
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve"); ax.legend()
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "06_precision_recall.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  ✓ {path}"); return path


# ---------------------------------------------------------------------------
# 7. FEATURE IMPORTANCE (PCA components + graph features)
# ---------------------------------------------------------------------------

def plot_feature_importance(model, feature_cols, top_n=15):
    importance = model.feature_importances_
    indices = np.argsort(importance)[-top_n:]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(range(top_n), importance[indices], color=COLORS["teal"])
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_cols[i] for i in indices])
    ax.set_xlabel("Importance (Gain)")
    ax.set_title("Top Features – XGBoost (PCA components + graph features)")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "07_feature_importance.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  ✓ {path}"); return path


# ---------------------------------------------------------------------------
# 8. TIME STEP ANALYSIS
# ---------------------------------------------------------------------------

def plot_timestep_analysis(df):
    """Illicit ratio per time step — shows temporal pattern."""
    ts_stats = df.groupby("time_step").agg(
        total=("is_illicit", "count"), illicit=("is_illicit", "sum")
    )
    ts_stats["rate"] = ts_stats["illicit"] / ts_stats["total"] * 100

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    ax1.bar(ts_stats.index, ts_stats["total"], color=COLORS["teal"], alpha=0.7)
    ax1.set_ylabel("Transactions"); ax1.set_title("Transactions & Illicit Rate per Time Step")

    ax2.plot(ts_stats.index, ts_stats["rate"], color=COLORS["red"], linewidth=2, marker="o", markersize=4)
    ax2.fill_between(ts_stats.index, ts_stats["rate"], alpha=0.1, color=COLORS["red"])
    ax2.set_ylabel("Illicit Rate (%)"); ax2.set_xlabel("Time Step (~2 weeks each)")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "08_timestep_analysis.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  ✓ {path}"); return path


# ---------------------------------------------------------------------------
# 9. DRIFT TIMELINE
# ---------------------------------------------------------------------------

def plot_drift_timeline(summary_path="data/simulated_months/simulation_summary.json"):
    if not os.path.exists(summary_path):
        print(f"  ⚠ No simulation summary. Run 'python main.py simulate' first.")
        return None

    with open(summary_path) as f:
        summary = json.load(f)
    months = [s["month"] for s in summary]
    fraud_rates = [s["fraud_rate_pct"] for s in summary]
    drift = [s["drift_detected"] for s in summary]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    ax1.plot(months, fraud_rates, color=COLORS["red"], linewidth=2.5, marker="o", markersize=8)
    ax1.fill_between(months, fraud_rates, alpha=0.1, color=COLORS["red"])
    ax1.set_ylabel("Fraud Rate (%)"); ax1.set_title("12-Month Simulation – Fraud Rate & MLOps Events")

    for s in summary:
        if s["retrained"]:
            ax1.axvline(x=s["month"], color=COLORS["teal"], alpha=0.3, linestyle="--")

    bar_colors = [COLORS["red"] if d else COLORS["green"] for d in drift]
    ax2.bar(months, [1]*len(months), color=bar_colors, width=0.6)
    ax2.set_yticks([]); ax2.set_xlabel("Month"); ax2.set_xticks(months)
    ax2.legend(handles=[
        mpatches.Patch(color=COLORS["red"], label="Drift"),
        mpatches.Patch(color=COLORS["green"], label="Stable"),
    ])
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "09_drift_timeline.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  ✓ {path}"); return path


# ---------------------------------------------------------------------------
# 10. METRICS CARD
# ---------------------------------------------------------------------------

def plot_metrics_card(metrics):
    fig, ax = plt.subplots(figsize=(8, 4)); ax.axis("off")
    names = ["Precision", "Recall", "F1 Score", "ROC-AUC"]
    vals = [metrics["precision"], metrics["recall"], metrics["f1"], metrics["auc"]]
    cols = [COLORS["teal"], COLORS["mint"], COLORS["amber"], COLORS["navy"]]
    for i, (n, v, c) in enumerate(zip(names, vals, cols)):
        x = 0.05 + i * 0.24
        ax.text(x+0.1, 0.7, f"{v:.4f}", fontsize=28, fontweight="bold", color=c, ha="center", transform=ax.transAxes)
        ax.text(x+0.1, 0.35, n, fontsize=13, color=COLORS["gray"], ha="center", transform=ax.transAxes)
        ax.add_patch(plt.Rectangle((x+0.02, 0.15), 0.16, 0.08, transform=ax.transAxes, facecolor=c, alpha=0.2))
        ax.add_patch(plt.Rectangle((x+0.02, 0.15), 0.16*v, 0.08, transform=ax.transAxes, facecolor=c, alpha=0.8))
    ax.set_title("Model Performance Summary", fontsize=16, fontweight="bold", pad=20)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "10_metrics_card.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  ✓ {path}"); return path


# ---------------------------------------------------------------------------
# GENERATE ALL
# ---------------------------------------------------------------------------

def generate_all():
    from main import load_elliptic, prepare_features
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

    print("\n" + "="*60)
    print("  GENERATING VISUALIZATIONS")
    print("="*60 + "\n")

    DATA_DIR = os.environ.get("DATA_DIR", "data")
    df, df_edges = load_elliptic(DATA_DIR)
    y = df["is_illicit"]

    plot_class_distribution(y)
    plot_timestep_analysis(df)

    if not os.path.exists("models/aml_model.pkl"):
        print("\n⚠ No model. Run 'python main.py train' first.")
        return

    model = joblib.load("models/aml_model.pkl")
    feature_cols = joblib.load("models/feature_cols.pkl")
    scaler = joblib.load("models/scaler.pkl")
    pca = joblib.load("models/pca.pkl")

    X, _, _ = prepare_features(df, scaler=scaler, pca=pca, fit=False)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_proba),
    }

    plot_pca_scatter(X, y, scaler, pca, df)
    plot_umap_scatter(df, y)
    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_proba)
    plot_precision_recall(y_test, y_proba)
    plot_feature_importance(model, feature_cols)
    plot_drift_timeline()
    plot_metrics_card(metrics)

    print(f"\n✓ All visualizations saved to '{OUTPUT_DIR}/'")
    return metrics


# ---------------------------------------------------------------------------
# INTERACTIVE DASHBOARD
# ---------------------------------------------------------------------------

def launch_dashboard():
    from main import load_elliptic
    DATA_DIR = os.environ.get("DATA_DIR", "data")
    df, _ = load_elliptic(DATA_DIR)
    total = len(df); illicit = (df["is_illicit"]==1).sum(); licit = total - illicit

    metrics_html = ""
    if os.path.exists("models/aml_model.pkl"):
        from main import prepare_features
        model = joblib.load("models/aml_model.pkl")
        scaler = joblib.load("models/scaler.pkl")
        pca_m = joblib.load("models/pca.pkl")
        X, _, _ = prepare_features(df, scaler=scaler, pca=pca_m, fit=False)
        y = df["is_illicit"]
        from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
        _, X_t, _, y_t = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        y_p = model.predict(X_t); y_pr = model.predict_proba(X_t)[:, 1]
        p, r, f, a = precision_score(y_t,y_p), recall_score(y_t,y_p), f1_score(y_t,y_p), roc_auc_score(y_t,y_pr)
        metrics_html = f"""<div class="metrics-row">
            <div class="mc"><div class="mv" style="color:#0D9488">{p:.4f}</div><div class="ml">Precision</div></div>
            <div class="mc"><div class="mv" style="color:#14B8A6">{r:.4f}</div><div class="ml">Recall</div></div>
            <div class="mc"><div class="mv" style="color:#F59E0B">{f:.4f}</div><div class="ml">F1 Score</div></div>
            <div class="mc"><div class="mv" style="color:#0F1B2D">{a:.4f}</div><div class="ml">ROC-AUC</div></div>
        </div>"""

    # Check for simulation
    sim_html = ""
    sp = "data/simulated_months/simulation_summary.json"
    if os.path.exists(sp):
        with open(sp) as f: summary = json.load(f)
        rows = ""
        for s in summary:
            db = f'<span class="br">DRIFT</span>' if s["drift_detected"] else '<span class="bg">STABLE</span>'
            rb = f'<span class="bt">YES</span>' if s["retrained"] else '<span class="bgy">NO</span>'
            rows += f"<tr><td>Month {s['month']}</td><td>{s['transactions']:,}</td><td>{s['fraud_rate_pct']:.2f}%</td><td>{db}</td><td>{rb}</td></tr>"
        sim_html = f"<h2>12-Month Simulation</h2><table><thead><tr><th>Month</th><th>Txns</th><th>Fraud %</th><th>Drift</th><th>Retrained</th></tr></thead><tbody>{rows}</tbody></table>"

    # Chart images
    charts = [("Class Distribution", "01"), ("Time Step Analysis", "08"), ("PCA Projection", "02"),
              ("UMAP Projection", "03"), ("Confusion Matrix", "04"), ("ROC Curve", "05"),
              ("Precision-Recall", "06"), ("Feature Importance", "07"), ("Metrics Summary", "10"),
              ("Drift Timeline", "09")]
    chart_html = ""
    for title, num in charts:
        png = f"visualizations/{num}_*.png"
        import glob
        files = glob.glob(f"visualizations/{num}_*")
        if files:
            chart_html += f'<div class="cc"><h3>{title}</h3><img src="{files[0]}"></div>'

    html = f"""<!DOCTYPE html><html><head><meta charset="UTF-8"><title>AML Bitcoin Dashboard</title>
<style>*{{margin:0;padding:0;box-sizing:border-box}}body{{font-family:'Segoe UI',sans-serif;background:#F0F4F8;color:#0F1B2D}}
.hd{{background:#0F1B2D;color:#fff;padding:2rem 3rem}}.hd h1{{font-size:1.8rem}}.hd p{{color:#64748B;font-size:.95rem}}
.ct{{max-width:1200px;margin:2rem auto;padding:0 2rem}}
h2{{color:#0F1B2D;margin:2rem 0 1rem;border-bottom:2px solid #0D9488;padding-bottom:.5rem}}
.sr{{display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;margin-bottom:2rem}}
.sc{{background:#fff;padding:1.5rem;border-radius:8px;text-align:center;box-shadow:0 2px 8px rgba(0,0,0,.08)}}
.sv{{font-size:2rem;font-weight:bold}}.sl{{color:#64748B;font-size:.85rem;margin-top:.3rem}}
.metrics-row{{display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;margin:1rem 0 2rem}}
.mc{{background:#fff;padding:1.2rem;border-radius:8px;text-align:center;box-shadow:0 2px 8px rgba(0,0,0,.08)}}
.mv{{font-size:1.8rem;font-weight:bold}}.ml{{color:#64748B;font-size:.85rem;margin-top:.3rem}}
.cg{{display:grid;grid-template-columns:repeat(2,1fr);gap:1.5rem;margin:1rem 0}}
.cc{{background:#fff;padding:1rem;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,.08)}}
.cc img{{width:100%;border-radius:4px}}.cc h3{{font-size:.95rem;color:#0D9488;margin-bottom:.5rem}}
table{{width:100%;border-collapse:collapse;background:#fff;border-radius:8px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,.08)}}
th{{background:#0F1B2D;color:#fff;padding:.8rem 1rem;text-align:left}}td{{padding:.6rem 1rem;border-bottom:1px solid #E2E8F0}}
.br{{background:#FEE2E2;color:#EF4444;padding:.2rem .6rem;border-radius:12px;font-size:.75rem;font-weight:bold}}
.bg{{background:#D1FAE5;color:#22C55E;padding:.2rem .6rem;border-radius:12px;font-size:.75rem;font-weight:bold}}
.bt{{background:#CCFBF1;color:#0D9488;padding:.2rem .6rem;border-radius:12px;font-size:.75rem;font-weight:bold}}
.bgy{{background:#F1F5F9;color:#64748B;padding:.2rem .6rem;border-radius:12px;font-size:.75rem;font-weight:bold}}
</style></head><body>
<div class="hd"><h1>AML Bitcoin Fraud Detection – Dashboard</h1><p>Elliptic Dataset · XGBoost + PCA · MLOps Pipeline</p></div>
<div class="ct">
<h2>Dataset</h2>
<div class="sr">
<div class="sc"><div class="sv" style="color:#0D9488">{total:,}</div><div class="sl">Labeled Transactions</div></div>
<div class="sc"><div class="sv" style="color:#22C55E">{licit:,}</div><div class="sl">Licit</div></div>
<div class="sc"><div class="sv" style="color:#EF4444">{illicit:,}</div><div class="sl">Illicit</div></div>
<div class="sc"><div class="sv" style="color:#F59E0B">{illicit/total*100:.2f}%</div><div class="sl">Illicit Ratio</div></div>
</div>
<h2>Model Performance</h2>{metrics_html}
<h2>Visualizations</h2><div class="cg">{chart_html}</div>
{sim_html}
</div></body></html>"""

    path = os.path.join(OUTPUT_DIR, "dashboard.html")
    with open(path, "w") as f: f.write(html)
    print(f"\n✓ Dashboard: {path}")
    import webbrowser; webbrowser.open(f"file://{os.path.abspath(path)}")
    return path


if __name__ == "__main__":
    generate_all()
    launch_dashboard()