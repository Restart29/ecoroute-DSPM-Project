"""
Rebuild every model artifact the Streamlit app needs, straight from
data/processed_emissions_v2.csv (no raw Olist files required).

    python src/build_artifacts.py

Writes:
    models/carbon_model.pkl      XGBoost pipeline (prep + model)
    models/nn_model.pkl          MLP pipeline (prep + model)
    models/shap_explainer.pkl    SHAP TreeExplainer for the XGBoost model
    models/kmeans_hubs.pkl       K-Means (k=5) on customer lat/lng
    models/benchmark_metrics.json  holdout + 5-fold CV metrics
and refreshes hub_cluster_id in the CSV so it matches the new K-Means model.
"""
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "processed_emissions_v2.csv")
if not os.path.exists(DATA):
    DATA = os.path.join(ROOT, "processed_emissions_v2.csv")
MODELS = os.path.join(ROOT, "models")
os.makedirs(MODELS, exist_ok=True)

NUM = ["price", "freight_value", "weight_tonnes", "volume_cm3", "distance_km", "actual_lead_time_days"]
CAT = ["product_category_name", "seller_state", "customer_state", "hub_cluster_id"]
TARGET = "co2_road_kg"
FAST = "--fast" in sys.argv  # smaller models for quick tests

df = pd.read_csv(DATA)
print(f"Loaded {len(df):,} rows from {DATA}")

# 1. K-Means hubs on customer coordinates (lat, lng order: the app relies on it)
kmeans = KMeans(n_clusters=5, n_init=10, random_state=42).fit(df[["cust_lat", "cust_lng"]].to_numpy())
df["hub_cluster_id"] = kmeans.labels_
df.to_csv(DATA, index=False)
joblib.dump(kmeans, os.path.join(MODELS, "kmeans_hubs.pkl"))
print("K-Means centroids (lat, lng):\n", np.round(kmeans.cluster_centers_, 3))

for c in CAT:
    df[c] = df[c].astype(str)

X, y = df[NUM + CAT], df[TARGET]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)


def make_prep():
    return ColumnTransformer([
        ("num", StandardScaler(), NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT),
    ])


try:
    from xgboost import XGBRegressor
    tree = XGBRegressor(n_estimators=150 if FAST else 400, max_depth=6, learning_rate=0.05,
                        subsample=0.9, colsample_bytree=0.9, random_state=42, n_jobs=-1)
    tree_name = "XGBoost"
except ImportError:  # fallback keeps the pipeline/SHAP interface identical
    from sklearn.ensemble import GradientBoostingRegressor
    tree = GradientBoostingRegressor(n_estimators=100 if FAST else 300, max_depth=6, random_state=42)
    tree_name = "GradientBoosting (xgboost not installed)"

xgb_pipe = Pipeline([("prep", make_prep()), ("model", tree)])
nn_pipe = Pipeline([("prep", make_prep()), ("model", MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=60 if FAST else 300,
                                                                  early_stopping=True, random_state=42))])

kf = KFold(n_splits=5, shuffle=True, random_state=42)
metrics = {}
for key, pipe in [("XGBoost", xgb_pipe), ("Neural_Network", nn_pipe)]:
    cv = cross_val_score(pipe, X_tr, y_tr, cv=kf, scoring="r2", n_jobs=1)
    pipe.fit(X_tr, y_tr)
    p = pipe.predict(X_te)
    metrics[key] = {
        "R2": float(r2_score(y_te, p)),
        "RMSE": float(np.sqrt(mean_squared_error(y_te, p))),
        "MAE": float(mean_absolute_error(y_te, p)),
        "CV_R2_Mean": float(cv.mean()),
        "CV_R2_Std": float(cv.std()),
    }
    print(key, {k: round(v, 4) for k, v in metrics[key].items()})

joblib.dump(xgb_pipe, os.path.join(MODELS, "carbon_model.pkl"))
joblib.dump(nn_pipe, os.path.join(MODELS, "nn_model.pkl"))
metrics["meta"] = {"tree_model": tree_name, "rows": int(len(df)), "test_size": 0.2}
with open(os.path.join(MODELS, "benchmark_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

try:
    import shap
    explainer = shap.TreeExplainer(xgb_pipe.named_steps["model"])
    joblib.dump(explainer, os.path.join(MODELS, "shap_explainer.pkl"))
    print("SHAP explainer saved.")
except ImportError:
    print("shap not installed: skipped shap_explainer.pkl (the app still runs).")

print("Done. Artifacts in", MODELS)
