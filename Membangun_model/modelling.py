from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = MODEL_DIR / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "breast_cancer_model.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"
PREDICTIONS_PATH = ARTIFACT_DIR / "test_predictions.csv"


def _dataset_dir() -> Path:
    candidates = [
        PROJECT_ROOT / "breast_cancer_preprocessing",
        PROJECT_ROOT / "preprocessing" / "breast_cancer_preprocessing",
    ]
    for candidate in candidates:
        if (candidate / "train_preprocessed.csv").exists() and (candidate / "test_preprocessed.csv").exists():
            return candidate
    raise FileNotFoundError("Folder breast_cancer_preprocessing tidak ditemukan.")


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    data_dir = _dataset_dir()
    train_df = pd.read_csv(data_dir / "train_preprocessed.csv")
    test_df = pd.read_csv(data_dir / "test_preprocessed.csv")

    X_train = train_df.drop(columns=["target"])
    y_train = train_df["target"]
    X_test = test_df.drop(columns=["target"])
    y_test = test_df["target"]
    return X_train, X_test, y_train, y_test


def evaluate_model(model: RandomForestClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }


def log_to_mlflow(model: RandomForestClassifier, metrics: dict[str, float]) -> None:
    try:
        import mlflow
        import mlflow.sklearn
    except ImportError:
        print("MLflow belum terpasang, artefak lokal tetap disimpan.")
        return

    mlflow.set_tracking_uri(f"file:{MODEL_DIR / 'mlruns'}")
    mlflow.set_experiment("breast-cancer-classification")
    with mlflow.start_run(run_name="random_forest_baseline"):
        mlflow.log_params(
            {
                "model": "RandomForestClassifier",
                "n_estimators": model.n_estimators,
                "max_depth": model.max_depth,
                "random_state": model.random_state,
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, artifact_path="model")
        mlflow.log_artifact(str(METRICS_PATH))
        mlflow.log_artifact(str(PREDICTIONS_PATH))


def train_model() -> tuple[RandomForestClassifier, dict[str, float]]:
    X_train, X_test, y_train, y_test = load_data()
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)

    ARTIFACT_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    predictions = X_test.copy()
    predictions["actual"] = y_test.values
    predictions["prediction"] = model.predict(X_test)
    predictions["benign_probability"] = model.predict_proba(X_test)[:, 1]
    predictions.to_csv(PREDICTIONS_PATH, index=False)

    log_to_mlflow(model, metrics)
    return model, metrics


if __name__ == "__main__":
    _, result = train_model()
    print("Model tersimpan di:", MODEL_PATH)
    for metric_name, metric_value in result.items():
        print(f"{metric_name}: {metric_value:.4f}")
