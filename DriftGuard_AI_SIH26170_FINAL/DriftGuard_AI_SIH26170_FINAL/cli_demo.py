from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from config import PARAMETERS
from inference import score_dataframe
from risk_engine import explain_row


def main():
    parameter = "leakage"
    df = pd.read_csv(ROOT / "data" / "demo_burn_in.csv")
    early = df[["component_id", "lot_id", f"{parameter}_0h", f"{parameter}_24h"]].copy()
    scored = score_dataframe(early, parameter, ROOT / "models")
    top = scored.sort_values("risk_score", ascending=False).head(10)
    cfg = PARAMETERS[parameter]

    print("\nDRIFTGUARD AI - ENHANCED CLI DEMO")
    print(f"Parameter: {cfg.label} ({cfg.unit})")
    print("Prototype thresholds only - not official ISRO limits.\n")
    cols = [
        "component_id", "lot_id", f"{parameter}_0h", f"{parameter}_24h",
        "predicted_168h", "prediction_lower_90", "prediction_upper_90",
        "risk_score", "risk_level", "early_reject", "recommendation"
    ]
    print(top[cols].to_string(index=False))

    row = top.iloc[0]
    print("\nWhy the highest-risk component was flagged:")
    for reason in explain_row(row, parameter):
        print(" -", reason)


if __name__ == "__main__":
    main()
