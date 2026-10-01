from __future__ import annotations

import numpy as np
import pandas as pd
from config import PARAMETERS


def anomaly_percentile(raw_score: pd.Series) -> pd.Series:
    # Higher = more anomalous. Rank-based normalization is robust and explainable.
    return raw_score.rank(pct=True).clip(0, 1)


def calculate_risk(df: pd.DataFrame, parameter: str) -> pd.DataFrame:
    cfg = PARAMETERS[parameter]
    out = df.copy()
    pred = out["predicted_168h"]
    v24 = out[f"{parameter}_24h"]

    future_slope = (pred - v24) / (168 - 24)
    slope_ratio = np.maximum(future_slope / max(cfg.safety_slope_per_hour, 1e-9), 0)
    slope_score = np.clip(slope_ratio / 2.0, 0, 1)
    limit_score = np.clip(pred / cfg.absolute_max, 0, 1.25) / 1.25
    anomaly_score = out["anomaly_percentile"].clip(0, 1)

    # Weighted toward anomaly + predicted drift. Prototype decision rule.
    risk = 100 * (0.45 * anomaly_score + 0.35 * slope_score + 0.20 * limit_score)
    out["future_slope_per_hour"] = future_slope
    out["risk_score"] = np.clip(risk, 0, 100).round(1)
    out["risk_level"] = pd.cut(
        out["risk_score"], bins=[-1, 35, 65, 100], labels=["Normal", "Warning", "Critical"]
    ).astype(str)
    out["early_reject"] = (
        (future_slope > cfg.safety_slope_per_hour)
        | (pred > cfg.absolute_max)
        | ((out["anomaly_percentile"] > 0.93) & (out["risk_score"] > 55))
    )
    out["recommendation"] = out.apply(recommendation_for_row, axis=1)
    return out


def recommendation_for_row(row: pd.Series) -> str:
    risk = float(row.get("risk_score", 0))
    reject = bool(row.get("early_reject", False))
    if reject or risk >= 65:
        return "Early reject / engineering review"
    if risk >= 35:
        return "Continue test with closer monitoring"
    return "Continue normal burn-in testing"


def explain_row(row: pd.Series, parameter: str) -> list[str]:
    cfg = PARAMETERS[parameter]
    reasons = []
    z24 = abs(float(row.get(f"{parameter}_z_24h_lot", 0)))
    z0 = abs(float(row.get(f"{parameter}_z_0h_lot", 0)))
    early_slope = float(row.get(f"{parameter}_slope_0_24", 0))

    if z24 >= 2.5:
        reasons.append(f"24h value is {z24:.1f} standard deviations from its lot average.")
    elif z0 >= 2.5:
        reasons.append(f"0h value is {z0:.1f} standard deviations from its lot average.")

    if float(row.get("anomaly_percentile", 0)) >= 0.93:
        pct = float(row.get("anomaly_percentile", 0)) * 100
        reasons.append(f"Early behaviour is more unusual than about {pct:.0f}% of this analysed batch.")

    if early_slope > cfg.safety_slope_per_hour:
        reasons.append(
            f"Observed 0h→24h drift ({early_slope:.4f} {cfg.unit}/h) already exceeds the prototype safety slope."
        )

    slope = float(row.get("future_slope_per_hour", 0))
    if slope > cfg.safety_slope_per_hour:
        reasons.append(
            f"Predicted 24h→168h drift ({slope:.4f} {cfg.unit}/h) exceeds the prototype safety slope "
            f"({cfg.safety_slope_per_hour:.4f} {cfg.unit}/h)."
        )

    pred = float(row.get("predicted_168h", 0))
    if pred > cfg.absolute_max:
        reasons.append(
            f"Predicted 168h value ({pred:.2f} {cfg.unit}) exceeds the prototype absolute limit "
            f"({cfg.absolute_max:.2f} {cfg.unit})."
        )
    elif pred > 0.8 * cfg.absolute_max:
        reasons.append("Predicted 168h value is close to the prototype absolute limit.")

    lo = row.get("prediction_lower_90", None)
    hi = row.get("prediction_upper_90", None)
    if lo is not None and hi is not None and pd.notna(lo) and pd.notna(hi):
        if float(hi) > cfg.absolute_max:
            reasons.append("The upper end of the estimated 90% prediction range crosses the prototype absolute limit.")

    if not reasons:
        reasons.append("Early readings and predicted drift are consistent with the reference population.")
    return reasons
