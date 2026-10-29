"""Melatih model dengan hyperparameter tuning dan manual logging MLflow."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import mlflow.data
import mlflow.sklearn
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "namadataset_preprocessing" / "telco_churn_preprocessing.csv"
MLRUNS_DIR = BASE_DIR / "mlruns"
ARTIFACT_DIR = BASE_DIR / "artifacts"
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


def tune_model(X_train: pd.DataFrame, y_train: pd.Series) -> GridSearchCV:
    """Mencari kombinasi hyperparameter terbaik dengan GridSearchCV."""
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
    }

    search = GridSearchCV(
        estimator=RandomForestClassifier(
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        param_grid=param_grid,
        scoring="f1",
        cv=5,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)

    return search


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


def save_confusion_matrix(model, X_test: pd.DataFrame, y_test: pd.Series) -> Path:
    """Menyimpan visualisasi confusion matrix."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACT_DIR / "confusion_matrix.png"

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_estimator(
        model,
        X_test,
        y_test,
        display_labels=["Tidak Churn", "Churn"],
        cmap="Blues",
        ax=ax,
    )
    ax.set_title("Confusion Matrix - Random Forest Tuning")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

    return path


def save_feature_importance(model, feature_names, top_n: int = 15) -> Path:
    """Menyimpan visualisasi fitur paling berpengaruh."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACT_DIR / "feature_importance.png"

    importance = pd.Series(model.feature_importances_, index=feature_names)
    importance = importance.sort_values(ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(x=importance.values, y=importance.index, ax=ax)
    ax.set_title(f"Top {top_n} Feature Importance - Random Forest Tuning")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

    return path


def main() -> None:
    mlflow.set_tracking_uri(MLRUNS_DIR.as_uri())
    mlflow.set_experiment("telco-churn-tuning")

    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)

    search = tune_model(X_train, y_train)
    best_model = search.best_estimator_

    train_metrics = evaluate_model(best_model, X_train, y_train)
    test_metrics = evaluate_model(best_model, X_test, y_test)

    confusion_path = save_confusion_matrix(best_model, X_test, y_test)
    importance_path = save_feature_importance(best_model, X_train.columns)
    report_path = ARTIFACT_DIR / "classification_report.txt"
    report_path.write_text(
        classification_report(y_test, best_model.predict(X_test)),
        encoding="utf-8",
    )
    cv_results_path = ARTIFACT_DIR / "cv_results.csv"
    pd.DataFrame(search.cv_results_).to_csv(cv_results_path, index=False)

    with mlflow.start_run(run_name="random_forest_gridsearch_manual"):
        dataset = mlflow.data.from_pandas(
            df,
            source=DATA_PATH.as_uri(),
            name="telco_churn_preprocessing",
        )
        mlflow.log_input(dataset, context="training")

        mlflow.log_params(
            {
                "model_type": "RandomForestClassifier",
                "class_weight": "balanced",
                "random_state": RANDOM_STATE,
                "cv_folds": 5,
                "scoring": "f1",
                "search_space": json.dumps(
                    {
                        "n_estimators": [100, 200],
                        "max_depth": [None, 10, 20],
                        "min_samples_split": [2, 5],
                    }
                ),
            }
        )
        mlflow.log_params({f"best_{key}": value for key, value in search.best_params_.items()})

        mlflow.log_metrics(
            {f"training_{key}": value for key, value in train_metrics.items()}
        )
        mlflow.log_metrics({f"test_{key}": value for key, value in test_metrics.items()})
        mlflow.log_metric("best_cv_score", search.best_score_)

        mlflow.log_artifact(str(confusion_path), artifact_path="evaluation")
        mlflow.log_artifact(str(importance_path), artifact_path="evaluation")
        mlflow.log_artifact(str(report_path), artifact_path="evaluation")
        mlflow.log_artifact(str(cv_results_path), artifact_path="evaluation")

        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            signature=mlflow.models.infer_signature(X_train, best_model.predict(X_train)),
            input_example=X_train.iloc[:5],
        )

        print(f"Best params : {search.best_params_}")
        print(f"Run ID      : {mlflow.active_run().info.run_id}")
        for key, value in test_metrics.items():
            print(f"{key:10s}: {value:.4f}")


if __name__ == "__main__":
    main()
