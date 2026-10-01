from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, IsolationForest, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, precision_recall_fscore_support, confusion_matrix, fbeta_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from config import PARAMETERS, RANDOM_STATE
from features import build_early_features


def evaluate_anomaly(y_true, pred_flag):
    p, r, f1, _ = precision_recall_fscore_support(y_true, pred_flag, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, pred_flag, labels=[0, 1]).ravel()
    f2 = fbeta_score(y_true, pred_flag, beta=2, zero_division=0)
    return {
        "precision": float(p), "recall": float(r), "f1": float(f1), "f2": float(f2),
        "false_negatives": int(fn), "true_positives": int(tp), "false_positives": int(fp), "true_negatives": int(tn)
    }


def train_parameter(df: pd.DataFrame, parameter: str, model_dir: Path):
    prepared, features = build_early_features(df, parameter)
    X = prepared[features]
    y_reg = prepared[f"{parameter}_168h"]
    y_cls = prepared["is_defective"].astype(int)

    train_idx, test_idx = train_test_split(
        np.arange(len(prepared)), test_size=0.25, random_state=RANDOM_STATE, stratify=y_cls
    )
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y_reg.iloc[train_idx], y_reg.iloc[test_idx]
    yc_test = y_cls.iloc[test_idx]

    contamination = PARAMETERS[parameter].contamination
    anomaly_candidates = {
        "isolation_forest": Pipeline([
            ("scale", StandardScaler()),
            ("model", IsolationForest(n_estimators=350, contamination=contamination, random_state=RANDOM_STATE, n_jobs=-1)),
        ]),
        "local_outlier_factor": Pipeline([
            ("scale", StandardScaler()),
            ("model", LocalOutlierFactor(n_neighbors=35, contamination=contamination, novelty=True)),
        ]),
        "one_class_svm": Pipeline([
            ("scale", StandardScaler()),
            ("model", OneClassSVM(kernel="rbf", gamma="scale", nu=contamination)),
        ]),
    }
    anomaly_results = {}
    trained_anomaly = {}
    for name, model in anomaly_candidates.items():
        model.fit(X_train)
        pred = (model.predict(X_test) == -1).astype(int)
        anomaly_results[name] = evaluate_anomaly(yc_test, pred)
        trained_anomaly[name] = model

    # F2 weights recall more than precision, matching the SIH emphasis on avoiding false negatives.
    best_anomaly_name = max(anomaly_results, key=lambda n: (anomaly_results[n]["f2"], anomaly_results[n]["recall"]))
    best_anomaly = trained_anomaly[best_anomaly_name]
    anomaly_metrics = anomaly_results[best_anomaly_name]

    regressors = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=350, max_depth=10, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "gradient_boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
    }

    reg_results = {}
    trained = {}
    residuals = {}
    for name, model in regressors.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        abs_err = np.abs(y_test.to_numpy() - pred)
        mae = mean_absolute_error(y_test, pred)
        reg_results[name] = {
            "mae": float(mae),
            "abs_error_q90": float(np.quantile(abs_err, 0.90)),
            "abs_error_q95": float(np.quantile(abs_err, 0.95)),
        }
        trained[name] = model
        residuals[name] = abs_err

    best_name = min(reg_results, key=lambda n: reg_results[n]["mae"])
    best_reg = trained[best_name]

    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_anomaly, model_dir / f"{parameter}_anomaly.joblib")
    joblib.dump(best_reg, model_dir / f"{parameter}_regressor.joblib")

    medians = {c: float(prepared[c].median()) for c in [f"{parameter}_0h", f"{parameter}_24h"]}
    metadata = {
        "parameter": parameter,
        "feature_columns": features,
        "reference_medians": medians,
        "regression_models": reg_results,
        "selected_regressor": best_name,
        "selected_regressor_error_abs_q90": reg_results[best_name]["abs_error_q90"],
        "selected_regressor_error_abs_q95": reg_results[best_name]["abs_error_q95"],
        "anomaly_models": anomaly_results,
        "selected_anomaly_model": best_anomaly_name,
        "anomaly_metrics": anomaly_metrics,
    }
    with open(model_dir / f"{parameter}_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/demo_burn_in.csv")
    parser.add_argument("--models", default="models")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    model_dir = Path(args.models)
    all_meta = {}
    comparison_rows = []
    for p in PARAMETERS:
        meta = train_parameter(df, p, model_dir)
        all_meta[p] = meta
        print(f"\n[{p}] regressor={meta['selected_regressor']} anomaly={meta['selected_anomaly_model']}")
        print("regression:", meta["regression_models"])
        print("anomaly:", meta["anomaly_models"])
        for model_name, m in meta["regression_models"].items():
            comparison_rows.append({"parameter": p, "task": "regression", "model": model_name, "MAE": m["mae"], "F2": np.nan, "Recall": np.nan})
        for model_name, m in meta["anomaly_models"].items():
            comparison_rows.append({"parameter": p, "task": "anomaly", "model": model_name, "MAE": np.nan, "F2": m["f2"], "Recall": m["recall"]})
    with open(model_dir / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(all_meta, f, indent=2)
    pd.DataFrame(comparison_rows).to_csv(model_dir / "model_comparison_full.csv", index=False)


if __name__ == "__main__":
    main()
