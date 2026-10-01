from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from features import build_early_features
from risk_engine import anomaly_percentile, calculate_risk


def _read_metadata(parameter: str, model_dir: Path) -> dict:
    with open(model_dir / f"{parameter}_metadata.json", encoding="utf-8") as f:
        return json.load(f)


def _load_serialized_bundle(parameter: str, model_dir: Path):
    anomaly = joblib.load(model_dir / f"{parameter}_anomaly.joblib")
    regressor = joblib.load(model_dir / f"{parameter}_regressor.joblib")
    meta = _read_metadata(parameter, model_dir)
    return anomaly, regressor, meta


def _rebuild_bundle_locally(parameter: str, model_dir: Path, original_error: Exception):
    """Rebuild a model with the user's installed sklearn if an old pickle is incompatible.

    The SIH prototype ships with synthetic training data, so recreating a model locally is
    safer than depending on private scikit-learn pickle internals across versions.
    """
    project_root = Path(__file__).resolve().parents[1]
    demo_data = project_root / "data" / "demo_burn_in.csv"

    if not demo_data.exists():
        raise RuntimeError(
            "The saved ML model is incompatible with this scikit-learn installation and "
            "the bundled demo training dataset could not be found for automatic rebuilding."
        ) from original_error

    try:
        from train_models import train_parameter

        df = pd.read_csv(demo_data)
        train_parameter(df, parameter, model_dir)
        return _load_serialized_bundle(parameter, model_dir)
    except Exception as rebuild_error:
        raise RuntimeError(
            f"Could not load or rebuild the '{parameter}' model. "
            "Run REBUILD_MODELS_WINDOWS.bat once, then restart DriftGuard."
        ) from rebuild_error


def load_bundle(parameter: str, model_dir: str | Path = "models"):
    model_dir = Path(model_dir)
    try:
        return _load_serialized_bundle(parameter, model_dir)
    except (ModuleNotFoundError, ImportError, AttributeError, ValueError, TypeError, EOFError) as exc:
        # Common cause: sklearn/joblib pickle internals changed between versions.
        # Rebuild just the affected parameter locally and retry automatically.
        return _rebuild_bundle_locally(parameter, model_dir, exc)


def score_dataframe(df: pd.DataFrame, parameter: str, model_dir: str | Path = "models") -> pd.DataFrame:
    anomaly, regressor, meta = load_bundle(parameter, model_dir)
    prepared, features = build_early_features(df, parameter, meta.get("reference_medians"))
    X = prepared[features]

    # Most sklearn novelty/outlier estimators: larger score_samples = more normal, so invert it.
    raw = -anomaly.score_samples(X)
    prepared["raw_anomaly_score"] = raw
    prepared["anomaly_percentile"] = anomaly_percentile(prepared["raw_anomaly_score"])
    prepared["is_anomaly"] = anomaly.predict(X) == -1
    prepared["predicted_168h"] = regressor.predict(X)

    # Empirical uncertainty band from held-out validation errors. This is an estimated
    # prediction range for the prototype, not a formal calibrated confidence interval.
    q90 = float(meta.get("selected_regressor_error_abs_q90", 0.0))
    prepared["prediction_lower_90"] = np.maximum(prepared["predicted_168h"] - q90, 0)
    prepared["prediction_upper_90"] = prepared["predicted_168h"] + q90
    prepared["prediction_range_half_width"] = q90

    prepared = calculate_risk(prepared, parameter)
    return prepared
