"""Melatih model Telco Customer Churn dengan MLflow autolog."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import mlflow
import mlflow.data
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "namadataset_preprocessing" / "telco_churn_preprocessing.csv"
MLRUNS_DIR = BASE_DIR / "mlruns"
TARGET_COLUMN = "Churn"
RANDOM_STATE = 42


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Memuat dataset hasil preprocessing."""
    return pd.read_csv(path)


def split_data(df: pd.DataFrame):
    """Membagi data menjadi data latih dan data uji secara stratified."""
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    """Melatih model Random Forest."""
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    return model


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Menghitung metrik evaluasi pada data uji."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }


def main() -> None:
    mlflow.set_tracking_uri(MLRUNS_DIR.as_uri())
    mlflow.set_experiment("telco-churn-basic")
    mlflow.sklearn.autolog()

    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)

    with mlflow.start_run(run_name="random_forest_autolog"):
        dataset = mlflow.data.from_pandas(
            df,
            source=DATA_PATH.as_uri(),
            name="telco_churn_preprocessing",
        )
        mlflow.log_input(dataset, context="training")

        model = train_model(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        mlflow.log_metrics({f"test_{key}": value for key, value in metrics.items()})

        print(f"Run ID : {mlflow.active_run().info.run_id}")
        for key, value in metrics.items():
            print(f"{key:10s}: {value:.4f}")


if __name__ == "__main__":
    main()
