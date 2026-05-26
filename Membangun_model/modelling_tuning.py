from __future__ import annotations

import json
from pathlib import Path

import joblib
from modelling import ARTIFACT_DIR, load_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold


MODEL_DIR = Path(__file__).resolve().parent
TUNED_MODEL_PATH = ARTIFACT_DIR / "breast_cancer_model_tuned.joblib"
TUNING_RESULT_PATH = ARTIFACT_DIR / "tuning_result.json"


def run_tuning() -> tuple[RandomForestClassifier, dict[str, object]]:
    X_train, X_test, y_train, y_test = load_data()

    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [4, 8, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = GridSearchCV(
        estimator=RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1),
        param_grid=param_grid,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    result: dict[str, object] = {
        "best_params": search.best_params_,
        "best_cv_f1": search.best_score_,
        "test_accuracy": accuracy_score(y_test, y_pred),
        "test_precision": precision_score(y_test, y_pred),
        "test_recall": recall_score(y_test, y_pred),
        "test_f1_score": f1_score(y_test, y_pred),
        "test_roc_auc": roc_auc_score(y_test, y_proba),
    }

    ARTIFACT_DIR.mkdir(exist_ok=True)
    joblib.dump(best_model, TUNED_MODEL_PATH)
    TUNING_RESULT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")

    try:
        import mlflow
        import mlflow.sklearn
    except ImportError:
        print("MLflow belum terpasang, hasil tuning tetap disimpan lokal.")
    else:
        mlflow.set_tracking_uri(f"file:{MODEL_DIR / 'mlruns'}")
        mlflow.set_experiment("breast-cancer-classification")
        with mlflow.start_run(run_name="random_forest_tuning"):
            mlflow.log_params(search.best_params_)
            mlflow.log_metrics({key: value for key, value in result.items() if isinstance(value, float)})
            mlflow.sklearn.log_model(best_model, artifact_path="model")
            mlflow.log_artifact(str(TUNING_RESULT_PATH))

    return best_model, result


if __name__ == "__main__":
    _, tuning_result = run_tuning()
    print("Model tuning tersimpan di:", TUNED_MODEL_PATH)
    print(json.dumps(tuning_result, indent=2))
