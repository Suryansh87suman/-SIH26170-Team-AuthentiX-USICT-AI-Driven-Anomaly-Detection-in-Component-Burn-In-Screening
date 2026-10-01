from __future__ import annotations

import numpy as np
import pandas as pd

EARLY_FEATURE_SUFFIXES = [
    "0h", "24h", "delta_0_24", "slope_0_24", "ratio_24_0", "z_0h_lot", "z_24h_lot"
]


def required_early_columns(parameter: str) -> list[str]:
    return ["component_id", "lot_id", f"{parameter}_0h", f"{parameter}_24h"]


def _safe_group_zscore(s: pd.Series) -> pd.Series:
    std = s.std(ddof=0)
    if pd.isna(std) or std < 1e-9:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s.mean()) / std


def build_early_features(df: pd.DataFrame, parameter: str, reference_medians: dict | None = None):
    cols = required_early_columns(parameter)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for {parameter}: {missing}")

    out = df.copy()
    c0 = f"{parameter}_0h"
    c24 = f"{parameter}_24h"

    # Median fill by lot, then global median fallback.
    for c in [c0, c24]:
        lot_med = out.groupby("lot_id")[c].transform("median")
        out[c] = out[c].fillna(lot_med)
        fallback = (reference_medians or {}).get(c, out[c].median())
        out[c] = out[c].fillna(fallback)

    out[f"{parameter}_delta_0_24"] = out[c24] - out[c0]
    out[f"{parameter}_slope_0_24"] = out[f"{parameter}_delta_0_24"] / 24.0
    out[f"{parameter}_ratio_24_0"] = out[c24] / np.clip(out[c0], 1e-9, None)

    out[f"{parameter}_z_0h_lot"] = out.groupby("lot_id")[c0].transform(_safe_group_zscore)
    out[f"{parameter}_z_24h_lot"] = out.groupby("lot_id")[c24].transform(_safe_group_zscore)

    feature_cols = [
        c0,
        c24,
        f"{parameter}_delta_0_24",
        f"{parameter}_slope_0_24",
        f"{parameter}_ratio_24_0",
        f"{parameter}_z_0h_lot",
        f"{parameter}_z_24h_lot",
    ]
    return out, feature_cols
