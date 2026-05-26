from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import joblib
import pandas as pd
from prometheus_client import Counter, Gauge, start_http_server
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "Membangun_model"
MODEL_PATH = MODEL_DIR / "artifacts" / "breast_cancer_model.joblib"
TEST_DATA_CANDIDATES = [
    MODEL_DIR / "breast_cancer_preprocessing" / "test_preprocessed.csv",
    PROJECT_ROOT / "breast_cancer_preprocessing" / "test_preprocessed.csv",
    PROJECT_ROOT / "preprocessing" / "breast_cancer_preprocessing" / "test_preprocessed.csv",
]

MODEL_ACCURACY = Gauge("model_accuracy", "Current model accuracy on test data")
MODEL_PRECISION = Gauge("model_precision", "Current model precision on test data")
MODEL_RECALL = Gauge("model_recall", "Current model recall on test data")
MODEL_F1_SCORE = Gauge("model_f1_score", "Current model F1 score on test data")
INFERENCE_LATENCY = Gauge("model_inference_latency_seconds", "Inference latency for the latest batch")
PREDICTION_COUNTER = Counter("model_predictions_total", "Total predictions served by the exporter")


def ensure_model_exists() -> None:
    if MODEL_PATH.exists():
        return

    sys.path.insert(0, str(MODEL_DIR))
    from modelling import train_model

    train_model()


def test_data_path() -> Path:
    for candidate in TEST_DATA_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("File test_preprocessed.csv tidak ditemukan.")


def update_metrics() -> dict[str, float]:
    ensure_model_exists()
    model = joblib.load(MODEL_PATH)
    test_df = pd.read_csv(test_data_path())
    X_test = test_df.drop(columns=["target"])
    y_test = test_df["target"]

    started = time.perf_counter()
    y_pred = model.predict(X_test)
    latency = time.perf_counter() - started

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "latency": latency,
    }

    MODEL_ACCURACY.set(metrics["accuracy"])
    MODEL_PRECISION.set(metrics["precision"])
    MODEL_RECALL.set(metrics["recall"])
    MODEL_F1_SCORE.set(metrics["f1_score"])
    INFERENCE_LATENCY.set(metrics["latency"])
    PREDICTION_COUNTER.inc(len(y_pred))
    return metrics


def run_exporter(port: int, interval: int) -> None:
    start_http_server(port)
    print(f"Prometheus exporter berjalan di http://localhost:{port}/metrics")
    while True:
        metrics = update_metrics()
        print(
            "accuracy={accuracy:.4f} precision={precision:.4f} "
            "recall={recall:.4f} f1={f1_score:.4f} latency={latency:.4f}s".format(**metrics)
        )
        time.sleep(interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exporter metrik model Breast Cancer untuk Prometheus.")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--interval", type=int, default=10)
    parser.add_argument("--once", action="store_true", help="Hitung metrik sekali lalu keluar.")
    args = parser.parse_args()

    if args.once:
        print(update_metrics())
    else:
        run_exporter(args.port, args.interval)
