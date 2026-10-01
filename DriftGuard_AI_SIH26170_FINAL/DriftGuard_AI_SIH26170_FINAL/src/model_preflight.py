from __future__ import annotations

from pathlib import Path

from config import PARAMETERS
from inference import load_bundle


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    model_dir = root / "models"
    print("Checking DriftGuard ML model compatibility...")
    for parameter in PARAMETERS:
        load_bundle(parameter, model_dir)
        print(f"  [OK] {parameter}")
    print("All models are compatible with this environment.")


if __name__ == "__main__":
    main()
