from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from config import PARAMETERS, RANDOM_STATE


def _generate_parameter(rng, n, lot_ids, base_mean, base_sd, slope_mean, slope_sd, defect_mask, defect_type):
    # Lot-to-lot shifts make dynamic within-lot comparison meaningful.
    unique_lots = np.unique(lot_ids)
    lot_shift = {lot: rng.normal(0, base_sd * 0.35) for lot in unique_lots}
    base = np.array([rng.normal(base_mean + lot_shift[lot], base_sd) for lot in lot_ids])
    slope = np.maximum(rng.normal(slope_mean, slope_sd, size=n), 0)

    values = {}
    for hour in [0, 24, 96, 168]:
        noise = rng.normal(0, base_sd * (0.07 + hour / 5000), size=n)
        values[hour] = base + slope * hour + noise

    # Defect families: high-but-still-legal early value, accelerated drift, unstable/noisy.
    idx_high = defect_mask & (defect_type == "high_relative")
    idx_drift = defect_mask & (defect_type == "accelerating_drift")
    idx_noisy = defect_mask & (defect_type == "unstable")

    if idx_high.any():
        # Strong relative anomaly while often staying below absolute max at 24h.
        bump = base_sd * rng.uniform(5.0, 8.0, idx_high.sum())
        for hour, scale in [(0, 0.55), (24, 0.8), (96, 1.0), (168, 1.15)]:
            values[hour][idx_high] += bump * scale

    if idx_drift.any():
        extra_slope = rng.uniform(slope_mean * 5 + 0.01, slope_mean * 12 + 0.03, idx_drift.sum())
        # Small early signal, then accelerated late drift.
        values[24][idx_drift] += extra_slope * 16
        values[96][idx_drift] += extra_slope * 96 * 1.35
        values[168][idx_drift] += extra_slope * 168 * 1.75

    if idx_noisy.any():
        vals = idx_noisy.sum()
        values[24][idx_noisy] += rng.normal(base_sd * 2.5, base_sd * 1.5, vals)
        values[96][idx_noisy] += rng.normal(base_sd * 4.0, base_sd * 2.0, vals)
        values[168][idx_noisy] += rng.normal(base_sd * 5.5, base_sd * 2.5, vals)

    for hour in values:
        values[hour] = np.maximum(values[hour], 0.001)
    return values


def generate_dataset(n_components: int = 1200, seed: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    lots = np.array([f"LOT-{x:02d}" for x in rng.integers(1, 9, size=n_components)])
    component_ids = np.array([f"CMP-{i:05d}" for i in range(1, n_components + 1)])

    defect_mask = rng.random(n_components) < 0.09
    defect_types = np.full(n_components, "normal", dtype=object)
    choices = rng.choice(["high_relative", "accelerating_drift", "unstable"], size=defect_mask.sum(), p=[0.35, 0.45, 0.20])
    defect_types[defect_mask] = choices

    df = pd.DataFrame({
        "component_id": component_ids,
        "lot_id": lots,
        "is_defective": defect_mask.astype(int),
        "defect_type": defect_types,
    })

    specs = {
        "leakage": (10.0, 1.3, 0.004, 0.004),
        "iddq": (4.2, 0.55, 0.0015, 0.0012),
        "delay": (12.0, 1.0, 0.0030, 0.0020),
    }

    for key, (base_mean, base_sd, slope_mean, slope_sd) in specs.items():
        vals = _generate_parameter(
            rng, n_components, lots, base_mean, base_sd, slope_mean, slope_sd, defect_mask, defect_types
        )
        for hour, arr in vals.items():
            df[f"{key}_{hour}h"] = np.round(arr, 4)

    # Small amount of missing early data to test preprocessing robustness.
    for key in PARAMETERS:
        for hour in [0, 24]:
            mask = rng.random(n_components) < 0.006
            df.loc[mask, f"{key}_{hour}h"] = np.nan

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1200)
    parser.add_argument("--out", default="data/demo_burn_in.csv")
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = generate_dataset(args.rows)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df):,} rows -> {out}")
    print(df.head().to_string(index=False))


if __name__ == "__main__":
    main()
