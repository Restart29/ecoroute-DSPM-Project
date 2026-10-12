import os
import json
import joblib
import pandas as pd
import numpy as np

import xgboost as xgb
import shap
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def train_and_benchmark(data_path="data/processed_emissions_v2.csv", model_dir="models"):
    print("=" * 60)
    print("🚀 EXECUTING MODEL TRAINING & BENCHMARKING PIPELINE")
    print("=" * 60)

    # 1. Load Preprocessed Data
    print(f"\n[1/5] Loading engineered dataset from: {data_path}")
    df = pd.read_csv(data_path)

    numeric_features = [
        "price", "freight_value", "weight_tonnes", 
        "volume_cm3", "distance_km", "actual_lead_time_days"
    ]
    categorical_features = [
        "product_category_name", "seller_state", 
        "customer_state", "hub_cluster_id"
    ]
    target = "co2_road_kg"

    X = df[numeric_features + categorical_features]
    y = df[target]

    # Convert cluster ID to string category so OneHotEncoder handles it properly
    X["hub_cluster_id"] = X["hub_cluster_id"].astype(str)

    # 80/20 Train-Test Holdout Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"Dataset split: {X_train.shape[0]} training samples, {X_test.shape[0]} holdout test samples.")

    # 2. Build Preprocessing Transformers
    # Neural Networks require standard scaling for numeric features; Tree models benefit as well
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)
        ]
    )

    # 3. Model Architecture Definitions
    print("\n[2/5] Initializing Candidate Model Architectures:")
    print("  • Model A: Gradient Boosted Trees (XGBoost Regressor)")
    print("  • Model B: Deep Multi-Layer Perceptron (Neural Network - 2 Hidden Layers: 64, 32)")

    xgb_model = xgb.XGBRegressor(
        n_estimators=180,
        learning_rate=0.07,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        n_jobs=-1
    )

    nn_model = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=0.001,
        max_iter=250,
        random_state=42,
        early_stopping=True
    )

    xgb_pipeline = Pipeline(steps=[("prep", preprocessor), ("model", xgb_model)])
    nn_pipeline = Pipeline(steps=[("prep", preprocessor), ("model", nn_model)])

    # 4. Rigorous 5-Fold Cross Validation Comparison
    print("\n[3/5] Running 5-Fold Cross-Validation on Training Set...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    xgb_cv_r2 = cross_val_score(xgb_pipeline, X_train, y_train, cv=kf, scoring="r2", n_jobs=-1)
    nn_cv_r2 = cross_val_score(nn_pipeline, X_train, y_train, cv=kf, scoring="r2", n_jobs=-1)

    print(f"  ➜ XGBoost 5-Fold R² Mean:       {np.mean(xgb_cv_r2):.4f} (+/- {np.std(xgb_cv_r2):.4f})")
    print(f"  ➜ Neural Network 5-Fold R² Mean: {np.mean(nn_cv_r2):.4f} (+/- {np.std(nn_cv_r2):.4f})")

    # 5. Fit & Test-Set Evaluation
    print("\n[4/5] Fitting final models on full training split...")
    xgb_pipeline.fit(X_train, y_train)
    nn_pipeline.fit(X_train, y_train)

    # Predictions
    xgb_preds = xgb_pipeline.predict(X_test)
    nn_preds = nn_pipeline.predict(X_test)

    # Metrics computation
    metrics = {
        "XGBoost": {
            "R2": float(r2_score(y_test, xgb_preds)),
            "RMSE": float(np.sqrt(mean_squared_error(y_test, xgb_preds))),
            "MAE": float(mean_absolute_error(y_test, xgb_preds)),
            "CV_R2_Mean": float(np.mean(xgb_cv_r2)),
            "CV_R2_Std": float(np.std(xgb_cv_r2))
        },
        "Neural_Network": {
            "R2": float(r2_score(y_test, nn_preds)),
            "RMSE": float(np.sqrt(mean_squared_error(y_test, nn_preds))),
            "MAE": float(mean_absolute_error(y_test, nn_preds)),
            "CV_R2_Mean": float(np.mean(nn_cv_r2)),
            "CV_R2_Std": float(np.std(nn_cv_r2))
        }
    }

    print("\n" + "=" * 45)
    print("📊 FINAL HOLDOUT TEST EVALUATION BENCHMARK")
    print("=" * 45)
    print(f"{'Metric':<12} | {'XGBoost Regressor':<18} | {'Neural Network (MLP)':<18}")
    print("-" * 55)
    print(f"{'R² Score':<12} | {metrics['XGBoost']['R2']:<18.4f} | {metrics['Neural_Network']['R2']:<18.4f}")
    print(f"{'RMSE (kg)':<12} | {metrics['XGBoost']['RMSE']:<18.4f} | {metrics['Neural_Network']['RMSE']:<18.4f}")
    print(f"{'MAE (kg)':<12} | {metrics['XGBoost']['MAE']:<18.4f} | {metrics['Neural_Network']['MAE']:<18.4f}")
    print("=" * 45)

    # 6. Save Artifacts for Real-Time Inference Dashboard
    print("\n[5/5] Serializing model weights and explainability engines...")
    os.makedirs(model_dir, exist_ok=True)

    joblib.dump(xgb_pipeline, os.path.join(model_dir, "carbon_model.pkl"))
    joblib.dump(nn_pipeline, os.path.join(model_dir, "nn_model.pkl"))

    # Save benchmark metrics to JSON for Tab 4 in app.py
    with open(os.path.join(model_dir, "benchmark_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    # Pre-compute SHAP TreeExplainer using a sampled training background
    print("Computing SHAP TreeExplainer for executive feature attribution...")
    fitted_preprocessor = xgb_pipeline.named_steps["prep"]
    fitted_xgb = xgb_pipeline.named_steps["model"]

    # Transform a reference background sample of 200 points
    background_sample = fitted_preprocessor.transform(X_train.iloc[:200])
    explainer = shap.TreeExplainer(fitted_xgb, background_sample)
    joblib.dump(explainer, os.path.join(model_dir, "shap_explainer.pkl"))

    print("\n✅ Pipeline complete. All artifacts saved in models/:")
    print("  • models/carbon_model.pkl")
    print("  • models/nn_model.pkl")
    print("  • models/shap_explainer.pkl")
    print("  • models/benchmark_metrics.json")

if __name__ == "__main__":
    train_and_benchmark()