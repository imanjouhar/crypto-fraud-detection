"""
AML Crypto Fraud Detection – Elliptic Bitcoin Dataset
Iman Jouhar | DLBDSMTP01

python main.py              Full pipeline (train → simulate → dashboard)
python main.py train        Train model
python main.py api          Start REST API + dashboard
python main.py simulate     12-month drift simulation
python main.py visualize    Generate result charts
python main.py dashboard    Open interactive dashboard
python main.py monitor      Launch MLflow UI
python main.py send         Stream test transactions to running API
python main.py test         Run automated test suite
python main.py tune         Auto-tune hyperparameters (Optuna)
python main.py demo         Quick train + single prediction
"""

import os, sys, json, time, csv
from functools import wraps
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_score, recall_score, f1_score, roc_auc_score
)
from xgboost import XGBClassifier
import mlflow, mlflow.xgboost
import joblib

# ── Configuration ──────────────────────────────────────────

DATA_DIR        = os.environ.get("DATA_DIR", "data")
MODEL_DIR       = "models"
N_PCA           = int(os.environ.get("N_PCA", "30"))
API_KEY         = os.environ.get("API_KEY", "aml-secret-key-2024")
DRIFT_THRESHOLD = 0.05
DRIFT_RATIO     = 0.3
N_RAW_FEATURES  = 165
DRIFT_FEATURES  = [f"pca_{i}" for i in range(10)] + ["time_step", "in_degree", "out_degree"]

os.makedirs(MODEL_DIR, exist_ok=True)


def banner(text):
    """Print a formatted section header to the console."""
    print(f"\n{'='*60}\n  {text}\n{'='*60}\n")


def check_dataset():
    """Verify that all three Elliptic CSV files are present in the data directory."""
    needed = ["elliptic_txs_features.csv", "elliptic_txs_classes.csv", "elliptic_txs_edgelist.csv"]
    missing = [f for f in needed if not os.path.exists(os.path.join(DATA_DIR, f))]
    if missing:
        print("Missing in data/:", ", ".join(missing))
        print("Download: https://www.kaggle.com/datasets/ellipticco/elliptic-data-set")
        sys.exit(1)


# ── Data Loading ───────────────────────────────────────────

def load_elliptic(data_dir=DATA_DIR):
    """
    Load and merge the three Elliptic CSV files into a single DataFrame.

    The features CSV has 167 columns with no header:
        Column 0: txId
        Column 1: time_step (1-49, each ~2 weeks)
        Columns 2-166: 165 anonymized transaction features

    Returns:
        df: DataFrame with features, labels, and graph metrics.
        df_edges: DataFrame with the raw edge list.
    """
    df_feat = pd.read_csv(os.path.join(data_dir, "elliptic_txs_features.csv"), header=None)
    df_feat.columns = ["txId", "time_step"] + [f"feat_{i}" for i in range(N_RAW_FEATURES)]

    df_cls = pd.read_csv(os.path.join(data_dir, "elliptic_txs_classes.csv"))
    df_cls.columns = ["txId", "class"]

    df_edges = pd.read_csv(os.path.join(data_dir, "elliptic_txs_edgelist.csv"))
    df_edges.columns = ["txId1", "txId2"]

    df = df_feat.merge(df_cls, on="txId", how="left")
    df = df[df["class"].isin(["1", "2", 1, 2])].copy()
    df["class"] = df["class"].astype(int)
    df["is_illicit"] = (df["class"] == 1).astype(int)

    out_deg = df_edges.groupby("txId1").size().rename("out_degree")
    in_deg  = df_edges.groupby("txId2").size().rename("in_degree")
    df = df.merge(out_deg, left_on="txId", right_index=True, how="left")
    df = df.merge(in_deg,  left_on="txId", right_index=True, how="left")
    df["out_degree"] = df["out_degree"].fillna(0).astype(int)
    df["in_degree"]  = df["in_degree"].fillna(0).astype(int)

    return df, df_edges


def prepare_features(df, n_pca=N_PCA, scaler=None, pca=None, fit=True):
    """
    Transform raw features into model-ready input using StandardScaler and PCA.

    Scales the 165 anonymized features to zero mean and unit variance, then
    reduces dimensionality via PCA. Appends time_step, in_degree, and
    out_degree as additional features.

    Returns:
        X_df: DataFrame with PCA components and graph features.
        scaler: Fitted StandardScaler instance.
        pca: Fitted PCA instance.
    """
    raw_cols = [f"feat_{i}" for i in range(N_RAW_FEATURES)]
    X_raw = df[raw_cols].values

    if fit:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_raw)
        pca = PCA(n_components=n_pca, random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        var = pca.explained_variance_ratio_.sum() * 100
        print(f"  PCA: {N_RAW_FEATURES} features -> {n_pca} components ({var:.1f}% variance retained)")
    else:
        X_scaled = scaler.transform(X_raw)
        X_pca = pca.transform(X_scaled)

    pca_cols = [f"pca_{i}" for i in range(n_pca)]
    X_df = pd.DataFrame(X_pca, columns=pca_cols, index=df.index)
    X_df["time_step"]  = df["time_step"].values
    X_df["in_degree"]  = df["in_degree"].values
    X_df["out_degree"] = df["out_degree"].values

    return X_df, scaler, pca


# ── Step 3: Train ──────────────────────────────────────────

def step_train():
    """
    Train an XGBoost classifier on the Elliptic Bitcoin dataset.

    Loads the dataset, engineers features via PCA, splits into
    train/test sets with stratification, trains XGBoost with
    class imbalance handling. Logs to MLflow and saves artifacts.

    Returns:
        Dictionary with precision, recall, f1, and auc scores.
    """
    banner("Step 3 – Train XGBoost")
    check_dataset()

    print("  Loading Elliptic dataset ...")
    df, _ = load_elliptic(DATA_DIR)
    n_illicit = (df["is_illicit"]==1).sum()
    n_licit   = (df["is_illicit"]==0).sum()
    print(f"  {len(df):,} labeled transactions | {n_illicit:,} illicit | {n_licit:,} licit | {df['is_illicit'].mean()*100:.2f}% illicit")

    print("  Feature engineering + PCA ...")
    X, scaler, pca = prepare_features(df, fit=True)
    y = df["is_illicit"]
    feature_cols = list(X.columns)

    print("  Splitting 70/15/15 (train/validation/test) ...")
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.176, random_state=42, stratify=y_trainval)
    print(f"  Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

    print("  Training ...")
    ratio = (y_train == 0).sum() / (y_train == 1).sum()
    model = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        scale_pos_weight=ratio, eval_metric="aucpr",
        use_label_encoder=False, random_state=42,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=20)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_proba)

    print("\n" + classification_report(y_test, y_pred, target_names=["Licit", "Illicit"]))
    print(f"  ROC-AUC: {auc:.4f}")
    print(f"  Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment("Crypto-Fraud-Detection")
    with mlflow.start_run(run_name="xgboost_elliptic_pca"):
        mlflow.log_param("n_estimators", 200)
        mlflow.log_param("max_depth", 6)
        mlflow.log_param("scale_pos_weight", round(ratio, 2))
        mlflow.log_param("pca_components", N_PCA)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", auc)
        mlflow.xgboost.log_model(model, "model")

    joblib.dump(model, os.path.join(MODEL_DIR, "aml_model.pkl"))
    joblib.dump(feature_cols, os.path.join(MODEL_DIR, "feature_cols.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(pca, os.path.join(MODEL_DIR, "pca.pkl"))
    print(f"\n  Saved to {MODEL_DIR}/")
    return {"precision": prec, "recall": rec, "f1": f1, "auc": auc}


# ── Step 4–5: REST API + Auth ──────────────────────────────

def step_api():
    """
    Start a Flask REST API serving the trained model.

    Endpoints:
        GET  /health         Health check (no auth).
        POST /predict        Score a single transaction (requires X-API-Key).
        POST /predict/batch  Score multiple transactions (requires X-API-Key).
    """
    banner("Step 4 – REST API")
    if not os.path.exists(os.path.join(MODEL_DIR, "aml_model.pkl")):
        print("No model found. Run: python main.py train"); sys.exit(1)

    from flask import Flask, request, jsonify, Response

    app = Flask(__name__)
    _model    = joblib.load(os.path.join(MODEL_DIR, "aml_model.pkl"))
    _features = joblib.load(os.path.join(MODEL_DIR, "feature_cols.pkl"))
    _scaler   = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    _pca      = joblib.load(os.path.join(MODEL_DIR, "pca.pkl"))

    def require_key(f):
        """Decorator that rejects requests without a valid API key."""
        @wraps(f)
        def decorated(*args, **kwargs):
            if request.headers.get("X-API-Key") != API_KEY:
                return jsonify({"error": "Unauthorized"}), 401
            return f(*args, **kwargs)
        return decorated

    def transform(data):
        """Apply scaler + PCA to raw input and return model-ready DataFrame."""
        raw = np.array([[data.get(f"feat_{i}", 0) for i in range(N_RAW_FEATURES)]])
        X_pca = _pca.transform(_scaler.transform(raw))
        df = pd.DataFrame(X_pca, columns=[f"pca_{i}" for i in range(X_pca.shape[1])])
        df["time_step"]  = data.get("time_step", 0)
        df["in_degree"]  = data.get("in_degree", 0)
        df["out_degree"] = data.get("out_degree", 0)
        for col in _features:
            if col not in df.columns:
                df[col] = 0
        return df[_features]

    @app.route("/health")
    def health():
        """Return API health status."""
        return jsonify({"status": "healthy"})

    @app.route("/dashboard")
    def dashboard():
        """Serve the interactive EDA dashboard with live data from model artifacts."""
        try:
            from visualize import build_dashboard_html
            return Response(build_dashboard_html(), mimetype="text/html")
        except Exception:
            static_path = os.path.join("visualizations", "dashboard.html")
            if os.path.exists(static_path):
                with open(static_path) as f:
                    return Response(f.read(), mimetype="text/html")
            return Response("<h1>Dashboard not ready. Run: python main.py train</h1>", mimetype="text/html")

    @app.route("/predict", methods=["POST"])
    @require_key
    def predict():
        """Score a single transaction and return illicit probability with risk level."""
        data = request.get_json()
        feat = transform(data)
        proba = _model.predict_proba(feat)[0][1]
        risk = "HIGH" if proba >= 0.7 else "MEDIUM" if proba >= 0.3 else "LOW"
        return jsonify({"is_illicit": int(proba >= 0.5), "probability": round(float(proba), 4), "risk_level": risk})

    @app.route("/predict/batch", methods=["POST"])
    @require_key
    def predict_batch():
        """Score multiple transactions in a single request."""
        txns = request.get_json().get("transactions", [])
        results = []
        for txn in txns:
            feat = transform(txn)
            proba = _model.predict_proba(feat)[0][1]
            results.append({"is_illicit": int(proba >= 0.5), "probability": round(float(proba), 4)})
        return jsonify({"predictions": results, "count": len(results)})

    print(f"  API:       http://localhost:5000/predict")
    print(f"  Dashboard: http://localhost:5000/dashboard")
    print(f"  Key: {API_KEY}")
    app.run(host="0.0.0.0", port=5000, debug=False)


# ── Drift Detection ───────────────────────────────────────

def save_baseline(X_df, path=os.path.join(MODEL_DIR, "baseline_stats.json")):
    """
    Save distribution statistics from current data as a drift detection baseline.

    Stores a sample of values for each monitored feature for later
    comparison using the Kolmogorov-Smirnov test.
    """
    baseline = {}
    for col in DRIFT_FEATURES:
        if col in X_df.columns:
            vals = X_df[col].dropna().tolist()[:5000]
            baseline[col] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "values": vals}
    with open(path, "w") as f:
        json.dump(baseline, f)
    print(f"  Baseline saved ({len(baseline)} features)")


def detect_drift(X_new, path=os.path.join(MODEL_DIR, "baseline_stats.json")):
    """
    Compare new data against baseline using the KS two-sample test.

    If p-value < 0.05 for a feature, it is considered drifted.
    If 30% or more of features drift, overall drift is declared.
    """
    with open(path) as f:
        baseline = json.load(f)
    drifted = []
    for col in DRIFT_FEATURES:
        if col not in X_new.columns or col not in baseline:
            continue
        _, p = stats.ks_2samp(np.array(baseline[col]["values"]), X_new[col].dropna().values)
        if p < DRIFT_THRESHOLD:
            drifted.append(col)
    ratio = len(drifted) / max(len(DRIFT_FEATURES), 1)
    return {"drift_detected": ratio >= DRIFT_RATIO, "drifted_features": drifted, "ratio": round(ratio, 2)}


def step_baseline():
    """Save drift detection baseline from the full training dataset."""
    banner("Save Drift Baseline")
    check_dataset()
    df, _ = load_elliptic(DATA_DIR)
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    pca = joblib.load(os.path.join(MODEL_DIR, "pca.pkl"))
    X, _, _ = prepare_features(df, scaler=scaler, pca=pca, fit=False)
    save_baseline(X)


# ── Step 6: 12-Month Simulation ───────────────────────────

def step_simulate():
    """
    Simulate 12 months of incoming data with progressive distribution drift.

    Maps the 49 Elliptic time steps into 12 monthly windows and applies
    increasing noise to simulate evolving laundering tactics:
        Months 1-3:   Stable baseline.
        Months 4-6:   Mild drift (small noise).
        Months 7-9:   Moderate drift (scaling + noise).
        Months 10-12: Strong drift (large scaling + label perturbation).

    Runs drift detection each month and retrains if threshold exceeded.
    """
    banner("Step 6 – 12-Month Drift Simulation")
    check_dataset()
    if not os.path.exists(os.path.join(MODEL_DIR, "baseline_stats.json")):
        step_baseline()

    out_dir = "data/simulated_months"
    os.makedirs(out_dir, exist_ok=True)

    df, _ = load_elliptic(DATA_DIR)
    scaler   = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    pca_obj  = joblib.load(os.path.join(MODEL_DIR, "pca.pkl"))
    ts       = sorted(df["time_step"].unique())
    per_m    = max(1, len(ts) // 12)
    summary  = []
    nf       = N_RAW_FEATURES

    for m in range(1, 13):
        print(f"\n  -- Month {m}/12 --")

        start = (m - 1) * per_m
        end   = m * per_m if m < 12 else len(ts)
        chunk = df[df["time_step"].isin(ts[start:end])].copy()
        if len(chunk) == 0:
            continue

        rng  = np.random.RandomState(42 + m)
        fcol = [f"feat_{i}" for i in range(nf)]

        if m <= 3:
            pass
        elif m <= 6:
            chunk[fcol] = chunk[fcol].values + rng.normal(0, 0.3, (len(chunk), nf))
        elif m <= 9:
            s = rng.uniform(0.8, 1.3, nf)
            chunk[fcol] = chunk[fcol].values * s + rng.normal(0, 0.8, (len(chunk), nf))
        else:
            s = rng.uniform(0.5, 2.0, nf)
            chunk[fcol] = chunk[fcol].values * s + rng.normal(0, 1.5, (len(chunk), nf))
            n_flip = max(1, int(len(chunk[chunk["is_illicit"]==0]) * 0.02))
            idx = chunk[chunk["is_illicit"]==0].sample(n=min(n_flip, len(chunk[chunk["is_illicit"]==0])), random_state=42+m).index
            chunk.loc[idx, "is_illicit"] = 1

        fraud = chunk["is_illicit"].mean() * 100
        X_m, _, _ = prepare_features(chunk, scaler=scaler, pca=pca_obj, fit=False)

        try:
            dr = detect_drift(X_m)
        except Exception:
            dr = {"drift_detected": False, "drifted_features": []}

        drifted  = dr["drift_detected"]
        retrain  = drifted or True # Monthly retrain or when drifted

        print(f"  {len(chunk):,} txns | {fraud:.2f}% fraud | drift={drifted} | retrain={retrain}")

        if retrain:
            step_train()
            save_baseline(X_m)
            scaler  = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
            pca_obj = joblib.load(os.path.join(MODEL_DIR, "pca.pkl"))

        summary.append({"month": m, "transactions": len(chunk), "fraud_rate_pct": round(fraud, 2),
                         "drift_detected": drifted, "retrained": retrain})

    with open(os.path.join(out_dir, "simulation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'Month':<8}{'Txns':<10}{'Fraud%':<10}{'Drift':<10}{'Retrain':<10}")
    print("-" * 48)
    for s in summary:
        print(f"{s['month']:<8}{s['transactions']:<10}{s['fraud_rate_pct']:<10}"
              f"{'YES' if s['drift_detected'] else '-':<10}{'YES' if s['retrained'] else '-':<10}")


# ── Step 5: MLflow ─────────────────────────────────────────

def step_monitor():
    """Launch the MLflow tracking UI on port 5001."""
    banner("Step 5 – MLflow")
    print("  http://localhost:5001")
    os.system("mlflow ui --port 5001")


# ── Visualizations ─────────────────────────────────────────

def step_visualize():
    """Generate all static PNG charts from trained model and dataset."""
    banner("Generate Visualizations")
    check_dataset()
    from visualize import generate_all
    generate_all()

def step_dashboard():
    """Generate charts and open the interactive HTML dashboard in a browser."""
    banner("Interactive Dashboard")
    check_dataset()
    from visualize import generate_all, launch_dashboard
    generate_all()
    launch_dashboard()


# ── Demo ───────────────────────────────────────────────────

def step_demo():
    """Run a quick end-to-end demo: train the model and make a single prediction."""
    banner("Demo – Train + Predict")
    step_train()
    step_baseline()
    model = joblib.load("models/aml_model.pkl")
    features = joblib.load("models/feature_cols.pkl")
    test = {col: 0 for col in features}
    test.update({"time_step": 25, "in_degree": 5, "out_degree": 3})
    df = pd.DataFrame([test])[features]
    p = model.predict_proba(df)[0][1]
    print(f"\n  Prediction: {p:.4f} ({'HIGH' if p>=0.7 else 'MEDIUM' if p>=0.3 else 'LOW'})")


def test_predict():
    """Send a test prediction request to the running API."""
    import requests
    txn = {f"feat_{i}": 0 for i in range(N_RAW_FEATURES)}
    txn.update({"time_step": 25, "in_degree": 3, "out_degree": 2})
    try:
        r = requests.post("http://localhost:5000/predict", json=txn, headers={"X-API-Key": API_KEY}, timeout=5)
        print(json.dumps(r.json(), indent=2))
    except Exception as e:
        print(f"API not running. Start with: python main.py api")


# ── Auto-tuning (Optuna) ───────────────────────────────────

def step_tune():
    """
    Automatically find the best hyperparameters using Optuna.

    Tests different combinations of n_estimators, max_depth, learning_rate,
    and PCA components. Saves the best parameters and retrains the model
    with them. Runs 30 trials (~5 minutes).
    """
    banner("Auto-tune – finding best hyperparameters")
    check_dataset()

    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("  Install optuna first: pip install optuna")
        return

    print("  Loading data ...")
    df, _ = load_elliptic(DATA_DIR)
    X, scaler, pca = prepare_features(df, fit=True)
    y = df["is_illicit"]
    feature_cols = list(X.columns)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    ratio = (y_train == 0).sum() / (y_train == 1).sum()

    def objective(trial):
        """Optuna objective: maximize F1 score."""
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "scale_pos_weight": ratio,
            "eval_metric": "aucpr",
            "use_label_encoder": False,
            "random_state": 42,
        }
        m = XGBClassifier(**params)
        m.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        pred = m.predict(X_test)
        return f1_score(y_test, pred)

    print("  Running 30 trials ...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30, show_progress_bar=True)

    best = study.best_params
    print(f"\n  Best F1: {study.best_value:.4f}")
    print(f"  Best params: {json.dumps(best, indent=2)}")

    # Save best params
    best_path = os.path.join(MODEL_DIR, "best_params.json")
    with open(best_path, "w") as f:
        json.dump(best, f, indent=2)
    print(f"  Saved to {best_path}")

    # Retrain with best params
    print("\n  Retraining with best params ...")
    best["scale_pos_weight"] = ratio
    best["eval_metric"] = "aucpr"
    best["use_label_encoder"] = False
    best["random_state"] = 42

    model = XGBClassifier(**best)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=20)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print("\n" + classification_report(y_test, y_pred, target_names=["Licit", "Illicit"]))
    print(f"  ROC-AUC: {auc:.4f}")

    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment("Crypto-Fraud-Detection")
    with mlflow.start_run(run_name="xgboost_tuned"):
        for k, v in best.items():
            mlflow.log_param(k, v)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", auc)

    joblib.dump(model, os.path.join(MODEL_DIR, "aml_model.pkl"))
    joblib.dump(feature_cols, os.path.join(MODEL_DIR, "feature_cols.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(pca, os.path.join(MODEL_DIR, "pca.pkl"))
    print(f"\n  Tuned model saved to {MODEL_DIR}/")
    return {"precision": prec, "recall": rec, "f1": f1, "auc": auc}


# ── Sender (stream transactions to API) ────────────────────

def step_send():
    """Simulate live Bitcoin transaction streaming to the running API."""
    banner("Sender – streaming transactions to API")
    import requests as req
    import csv, time

    log_path = "visualizations/predictions_log.csv"
    os.makedirs("visualizations", exist_ok=True)

    # Generate test transactions
    txns = []
    data_path = os.path.join(DATA_DIR, "elliptic_txs_features.csv")
    if os.path.exists(data_path):
        df = pd.read_csv(data_path, header=None, nrows=30)
        df.columns = ["txId", "time_step"] + [f"feat_{i}" for i in range(N_RAW_FEATURES)]
        for _, row in df.iterrows():
            txn = {f"feat_{i}": round(float(row[f"feat_{i}"]), 3) for i in range(N_RAW_FEATURES)}
            txn.update({"time_step": int(row["time_step"]), "in_degree": int(np.random.randint(0, 10)), "out_degree": int(np.random.randint(0, 10))})
            txns.append(txn)
    else:
        for _ in range(20):
            txn = {f"feat_{i}": round(float(np.random.randn()), 3) for i in range(N_RAW_FEATURES)}
            txn.update({"time_step": 25, "in_degree": 3, "out_degree": 2})
            txns.append(txn)

    log = open(log_path, "w", newline="")
    writer = csv.writer(log)
    writer.writerow(["index", "is_illicit", "probability", "risk_level", "ms"])
    headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}
    flagged = 0

    for i, txn in enumerate(txns):
        try:
            t0 = time.time()
            r = req.post("http://localhost:5000/predict", json=txn, headers=headers, timeout=10)
            ms = round((time.time() - t0) * 1000, 1)
            if r.status_code == 200:
                res = r.json()
                writer.writerow([i, res["is_illicit"], res["probability"], res["risk_level"], ms])
                if res["is_illicit"]: flagged += 1
                print(f"  [{i+1:>3}/{len(txns)}] {'ILLICIT' if res['is_illicit'] else 'licit':>7} | prob={res['probability']:.3f} | {res['risk_level']:<6} | {ms}ms")
        except req.ConnectionError:
            print(f"  API not running. Start with: python main.py api"); break
        time.sleep(0.2)

    log.close()
    print(f"\n  {len(txns)} sent, {flagged} flagged. Log: {log_path}")


# ── Test suite ─────────────────────────────────────────────

def step_test():
    """Run automated tests verifying all project artifacts."""
    banner("Test Suite")
    passed, failed = 0, 0

    def check(name, ok):
        nonlocal passed, failed
        if ok: passed += 1; print(f"  PASS  {name}")
        else: failed += 1; print(f"  FAIL  {name}")

    # Model artifacts
    print("\n  -- Model --")
    for f in ["models/aml_model.pkl", "models/feature_cols.pkl", "models/scaler.pkl", "models/pca.pkl"]:
        check(f, os.path.exists(f))
    if os.path.exists("models/aml_model.pkl"):
        m = joblib.load("models/aml_model.pkl")
        check("Model has predict_proba", hasattr(m, "predict_proba"))

    # Baseline
    print("\n  -- Drift baseline --")
    bp = "models/baseline_stats.json"
    check("Baseline exists", os.path.exists(bp))
    if os.path.exists(bp):
        with open(bp) as f: d = json.load(f)
        check(f"Baseline has {len(d)} features", len(d) > 0)

    # Visualizations
    print("\n  -- Outputs --")
    for f in ["visualizations/dashboard.html", "visualizations/01_class_distribution.png", "visualizations/03_roc_curve.png"]:
        check(f, os.path.exists(f))

    # Simulation
    print("\n  -- Simulation --")
    sp = "data/simulated_months/simulation_summary.json"
    check("Simulation summary", os.path.exists(sp))
    if os.path.exists(sp):
        with open(sp) as f: s = json.load(f)
        check(f"{len(s)} months simulated", len(s) == 12)

    # CI/CD
    print("\n  -- Infrastructure --")
    check("GitHub Actions workflow", os.path.exists(".github/workflows/retrain.yml"))
    check("Dockerfile", os.path.exists("Dockerfile"))
    check("docker-compose.yml", os.path.exists("docker-compose.yml"))

    # API (if running)
    print("\n  -- API --")
    try:
        import requests as req
        r = req.get("http://localhost:5000/health", timeout=2)
        check("API /health responds", r.status_code == 200)
        r2 = req.post("http://localhost:5000/predict", json={}, timeout=2)
        check("API rejects no auth (401)", r2.status_code == 401)
    except Exception:
        print("  SKIP  API not running")

    print(f"\n  Results: {passed} passed, {failed} failed")


# ── Full Pipeline ──────────────────────────────────────────

def run_full():
    """Execute the complete MLOps pipeline: train, baseline, simulate, visualize."""
    banner("Crypto Fraud Detection – Full Pipeline")
    step_train()
    step_baseline()
    step_simulate()
    step_dashboard()
    banner("Done")
    print("  python main.py api        Start REST API")
    print("  python main.py send       Stream test transactions to API")
    print("  python main.py test       Run automated test suite")
    print("  python main.py monitor    Open MLflow UI")


# ── CLI ────────────────────────────────────────────────────

COMMANDS = {
    "train": step_train, "api": step_api, "baseline": step_baseline,
    "simulate": step_simulate, "monitor": step_monitor, "demo": step_demo,
    "predict": test_predict, "visualize": step_visualize, "dashboard": step_dashboard,
    "send": step_send, "test": step_test, "tune": step_tune,
}

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else None
    if cmd is None:
        run_full()
    elif cmd in COMMANDS:
        COMMANDS[cmd]()
    elif cmd in ("-h", "--help", "help"):
        print(__doc__)
    else:
        print(f"Unknown: '{cmd}'. Available: {', '.join(COMMANDS)}")
        sys.exit(1)
