from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd


@dataclass
class QualityIssue:
    severity: str
    check: str
    details: str
    affected_rows: int = 0

    def to_dict(self):
        return asdict(self)


def assess_data_quality(df: pd.DataFrame, parameters: list[str] | None = None) -> dict:
    parameters = parameters or ["leakage", "iddq", "delay"]
    issues: list[QualityIssue] = []
    required_base = ["component_id", "lot_id"]

    for col in required_base:
        if col not in df.columns:
            issues.append(QualityIssue("Critical", "Missing required column", f"Column '{col}' is missing.", len(df)))

    if "component_id" in df.columns:
        blank_ids = df["component_id"].isna().sum() + (df["component_id"].astype(str).str.strip() == "").sum()
        if blank_ids:
            issues.append(QualityIssue("Critical", "Missing component IDs", f"{blank_ids} rows have a missing/blank component_id.", int(blank_ids)))
        dupes = int(df["component_id"].duplicated(keep=False).sum())
        if dupes:
            issues.append(QualityIssue("Warning", "Duplicate component IDs", f"{dupes} rows share a duplicated component_id.", dupes))

    if "lot_id" in df.columns:
        missing_lot = int(df["lot_id"].isna().sum())
        if missing_lot:
            issues.append(QualityIssue("Warning", "Missing lot IDs", f"{missing_lot} rows have no lot_id.", missing_lot))

    checked_numeric = 0
    total_missing = 0
    negative_count = 0
    extreme_jump_count = 0

    for p in parameters:
        present = [c for c in [f"{p}_0h", f"{p}_24h", f"{p}_96h", f"{p}_168h"] if c in df.columns]
        if not present:
            continue
        checked_numeric += len(present)
        for c in present:
            numeric = pd.to_numeric(df[c], errors="coerce")
            non_numeric = int((df[c].notna() & numeric.isna()).sum())
            if non_numeric:
                issues.append(QualityIssue("Critical", "Non-numeric readings", f"{non_numeric} values in '{c}' cannot be parsed as numbers.", non_numeric))
            miss = int(numeric.isna().sum())
            total_missing += miss
            if miss:
                severity = "Warning" if c.endswith("0h") or c.endswith("24h") else "Info"
                issues.append(QualityIssue(severity, "Missing readings", f"'{c}' has {miss} missing readings.", miss))
            neg = int((numeric < 0).sum())
            negative_count += neg
            if neg:
                issues.append(QualityIssue("Critical", "Negative readings", f"'{c}' contains {neg} negative values; verify units/sensor export.", neg))

        c0, c24 = f"{p}_0h", f"{p}_24h"
        if c0 in df.columns and c24 in df.columns:
            a = pd.to_numeric(df[c0], errors="coerce")
            b = pd.to_numeric(df[c24], errors="coerce")
            base = a.abs().clip(lower=1e-9)
            jump = ((b - a).abs() / base) > 5.0
            n_jump = int(jump.fillna(False).sum())
            extreme_jump_count += n_jump
            if n_jump:
                issues.append(QualityIssue("Warning", "Extreme 0h→24h jumps", f"{n_jump} '{p}' rows change by more than 500%; verify sensor/unit consistency.", n_jump))

    if checked_numeric == 0:
        issues.append(QualityIssue("Critical", "No burn-in columns", "No supported parameter columns were found.", len(df)))

    # Transparent quality score: deductions are capped so one bad check cannot make it negative.
    severity_weights = {"Critical": 18, "Warning": 7, "Info": 2}
    deduction = sum(severity_weights.get(i.severity, 0) for i in issues)
    score = max(0, 100 - min(100, deduction))
    if not issues:
        score = 100

    return {
        "score": int(score),
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "issue_count": len(issues),
        "issues": [i.to_dict() for i in issues],
        "summary": {
            "missing_numeric_values": int(total_missing),
            "negative_values": int(negative_count),
            "extreme_early_jumps": int(extreme_jump_count),
        },
    }
