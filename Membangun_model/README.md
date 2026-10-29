# Membangun Model — Telco Customer Churn (fauzi19)

Folder ini berisi kode pelatihan model klasifikasi churn pelanggan Telco menggunakan
Random Forest dan MLflow Tracking.

## Struktur Folder

```
Membangun_model/
|-- modelling.py              # Pelatihan baseline + MLflow autolog
|-- modelling_tuning.py       # Hyperparameter tuning + manual logging + artefak
|-- requirements.txt
|-- screenshoot_dashboard.jpg # Bukti dashboard MLflow
|-- screenshoot_artifak.jpg   # Bukti artefak model di MLflow
|-- artifacts/                # Artefak evaluasi hasil tuning
|   |-- classification_report.txt
|   |-- confusion_matrix.png
|   |-- cv_results.csv
|   `-- feature_importance.png
`-- namadataset_preprocessing/
    `-- telco_churn_preprocessing.csv
```

## Dataset

Dataset hasil preprocessing dari IBM Telco Customer Churn: 7.021 baris, 24 kolom,
dengan target `Churn`. Data dibagi menjadi 80% data latih dan 20% data uji secara
stratified (`random_state=42`).

## Cara Menjalankan

```bash
pip install -r requirements.txt

# Baseline (autolog)
python modelling.py

# Hyperparameter tuning (GridSearchCV, 5-fold, scoring F1)
python modelling_tuning.py

# Melihat dashboard MLflow
mlflow ui --backend-store-uri mlruns
```

## Hasil Evaluasi

Model terbaik: `RandomForestClassifier` dengan `n_estimators=200`, `max_depth=10`,
`min_samples_split=2`, dan `class_weight=balanced`.

| Metrik | Nilai |
|---|---|
| Accuracy | 0.7665 |
| Precision | 0.5431 |
| Recall | 0.7446 |
| F1-score | 0.6281 |
| ROC-AUC | 0.8407 |
| Best CV F1 | 0.6323 |

Run ID model tuning: `94e9087c03c24c5c9eeaa6e98f0758ba`.

## Bukti

- `screenshoot_dashboard.jpg` — dashboard MLflow berisi run baseline dan tuning
  beserta metrik dan parameternya.
- `screenshoot_artifak.jpg` — artefak model (`model.pkl`), classification report,
  confusion matrix, feature importance, dan `cv_results.csv` pada MLflow.
- Folder `artifacts/` — salinan artefak evaluasi yang dihasilkan
  `modelling_tuning.py`.
