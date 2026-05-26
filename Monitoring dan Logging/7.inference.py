from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "Membangun_model"
MODEL_PATH = MODEL_DIR / "artifacts" / "breast_cancer_model.joblib"
DEFAULT_INPUT_PATH = PROJECT_ROOT / "breast_cancer_preprocessing" / "test_preprocessed.csv"


def ensure_model_exists() -> None:
    if MODEL_PATH.exists():
        return

    sys.path.insert(0, str(MODEL_DIR))
    from modelling import train_model

    train_model()


def run_inference(input_path: Path, rows: int) -> list[dict[str, object]]:
    ensure_model_exists()
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(input_path).head(rows)
    X = df.drop(columns=["target"], errors="ignore")

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    result: list[dict[str, object]] = []
    for index, (prediction, probability) in enumerate(zip(predictions, probabilities), start=1):
        result.append(
            {
                "row": index,
                "prediction": int(prediction),
                "prediction_label": "benign" if int(prediction) == 1 else "malignant",
                "benign_probability": round(float(probability), 6),
            }
        )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference model klasifikasi Breast Cancer.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--rows", type=int, default=5)
    args = parser.parse_args()

    print(json.dumps(run_inference(args.input, args.rows), indent=2))
