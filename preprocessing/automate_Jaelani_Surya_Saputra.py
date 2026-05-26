from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "breast_cancer_raw"
PREPROCESSED_DIR = PROJECT_ROOT / "preprocessing" / "breast_cancer_preprocessing"
RAW_PATH = RAW_DIR / "breast_cancer_raw.csv"
TRAIN_PATH = PREPROCESSED_DIR / "train_preprocessed.csv"
TEST_PATH = PREPROCESSED_DIR / "test_preprocessed.csv"


def load_raw_dataset() -> pd.DataFrame:
    """Load Breast Cancer Wisconsin dataset and keep a raw CSV copy."""
    RAW_DIR.mkdir(exist_ok=True)

    dataset = load_breast_cancer(as_frame=True)
    df = dataset.frame.copy()
    df["target_name"] = df["target"].map({0: "malignant", 1: "benign"})
    df.to_csv(RAW_PATH, index=False)

    return df


def preprocess_dataset(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean, split, impute, scale, and return ready-to-train data."""
    clean_df = df.drop_duplicates().copy()

    X = clean_df.drop(columns=["target", "target_name"])
    y = clean_df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    preprocessing_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    X_train_processed = preprocessing_pipeline.fit_transform(X_train)
    X_test_processed = preprocessing_pipeline.transform(X_test)

    train_processed = pd.DataFrame(
        X_train_processed,
        columns=X.columns,
        index=X_train.index,
    )
    train_processed["target"] = y_train.values

    test_processed = pd.DataFrame(
        X_test_processed,
        columns=X.columns,
        index=X_test.index,
    )
    test_processed["target"] = y_test.values

    return train_processed, test_processed


def save_preprocessed_data(
    train_processed: pd.DataFrame,
    test_processed: pd.DataFrame,
) -> None:
    """Save processed train and test datasets."""
    PREPROCESSED_DIR.mkdir(exist_ok=True)
    train_processed.to_csv(TRAIN_PATH, index=False)
    test_processed.to_csv(TEST_PATH, index=False)


def run_preprocessing() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the complete preprocessing workflow."""
    raw_df = load_raw_dataset()
    train_processed, test_processed = preprocess_dataset(raw_df)
    save_preprocessed_data(train_processed, test_processed)
    return train_processed, test_processed


if __name__ == "__main__":
    train_data, test_data = run_preprocessing()
    print(f"Raw dataset saved to: {RAW_PATH}")
    print(f"Train dataset saved to: {TRAIN_PATH} with shape {train_data.shape}")
    print(f"Test dataset saved to: {TEST_PATH} with shape {test_data.shape}")
