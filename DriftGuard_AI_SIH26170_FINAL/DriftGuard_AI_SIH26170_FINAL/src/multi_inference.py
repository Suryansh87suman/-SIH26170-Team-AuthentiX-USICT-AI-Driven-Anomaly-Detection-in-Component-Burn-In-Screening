from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from inference import score_dataframe


def supported_parameters(df: pd.DataFrame) -> list[str]:
    out = []
    for p in ["leakage", "iddq", "delay"]:
        required = {"component_id", "lot_id", f"{p}_0h", f"{p}_24h"}
        if required.issubset(df.columns):
            out.append(p)
    return out


def score_all_parameters(df: pd.DataFrame, model_dir: str | Path = "models") -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    params = supported_parameters(df)
    if not params:
        raise ValueError("No complete supported early parameter set found.")

    frames = {}
    combined = df[["component_id", "lot_id"]].copy()
    risk_cols = []
    reject_cols = []
    for p in params:
        s = score_dataframe(df, p, model_dir)
        frames[p] = s
        combined[f"{p}_risk"] = s["risk_score"].to_numpy()
        combined[f"{p}_level"] = s["risk_level"].to_numpy()
        combined[f"{p}_predicted_168h"] = s["predicted_168h"].to_numpy()
        combined[f"{p}_early_reject"] = s["early_reject"].to_numpy()
        risk_cols.append(f"{p}_risk")
        reject_cols.append(f"{p}_early_reject")

    mean_risk = combined[risk_cols].mean(axis=1)
    max_risk = combined[risk_cols].max(axis=1)
    # Gives weight to overall degradation while not hiding one very risky parameter.
    combined["combined_risk_score"] = np.clip(0.60 * mean_risk + 0.40 * max_risk, 0, 100).round(1)
    combined["combined_risk_level"] = pd.cut(
        combined["combined_risk_score"], [-1, 35, 65, 100], labels=["Normal", "Warning", "Critical"]
    ).astype(str)
    combined["combined_early_reject"] = combined[reject_cols].any(axis=1)
    combined["dominant_risk_parameter"] = combined[risk_cols].idxmax(axis=1).str.replace("_risk", "", regex=False)
    return combined, frames
