# Machine Learning Operations (MLOps) — Telco Customer Churn

Proyek akhir MLOps yang mencakup pelatihan model klasifikasi churn pelanggan telekomunikasi, pelacakan eksperimen dengan MLflow, serving model, serta monitoring dan alerting dengan Prometheus dan Grafana.

## Isi Repository

| Folder / Berkas | Keterangan |
|---|---|
| `Membangun_model/` | Pelatihan model Random Forest, hyperparameter tuning, dan pelacakan dengan MLflow |
| `Monitoring dan Logging/` | Serving model, exporter metrik Prometheus, dashboard dan alerting Grafana, serta pengirim trafik inferensi |
| `Eksperimen_SML_fauzi19.txt` | Tautan repository eksperimen |
| `Workflow-CI.txt` | Tautan repository workflow CI |

## Dataset

Dataset hasil preprocessing dari IBM Telco Customer Churn: 7.021 baris, 24 kolom, dengan target `Churn`. Data dibagi 80% latih dan 20% uji secara stratified (`random_state=42`).

## Bagian 1 — Membangun Model

Berkas utama:
- `modelling.py`: pelatihan baseline dengan MLflow autolog.
- `modelling_tuning.py`: hyperparameter tuning (GridSearchCV, 5-fold, scoring F1) dengan manual logging dan penyimpanan artefak.

Cara menjalankan:

```bash
cd Membangun_model
pip install -r requirements.txt

# Baseline (autolog)
python modelling.py

# Hyperparameter tuning
python modelling_tuning.py

# Dashboard MLflow
mlflow ui --backend-store-uri mlruns
```

Hasil model terbaik (`RandomForestClassifier`, `n_estimators=200`, `max_depth=10`, `min_samples_split=2`, `class_weight=balanced`):

| Metrik | Nilai |
|---|---|
| Accuracy | 0.7665 |
| Precision | 0.5431 |
| Recall | 0.7446 |
| F1-score | 0.6281 |
| ROC-AUC | 0.8407 |
| Best CV F1 | 0.6323 |

Run ID model tuning: `94e9087c03c24c5c9eeaa6e98f0758ba`.

Artefak evaluasi tersedia di `Membangun_model/artifacts/` (classification report, confusion matrix, feature importance, dan `cv_results.csv`).

## Bagian 2 — Monitoring dan Logging

Model dilayani dengan `mlflow models serve`, lalu exporter Prometheus membungkus layanan tersebut dan mengekspos 10 metrik:

- `ml_predict_requests_total{status}`
- `ml_predict_latency_seconds`
- `ml_predictions_total{prediction}`
- `ml_predict_in_progress`
- `ml_request_batch_size`
- `ml_predict_errors_total{error_type}`
- `ml_last_request_timestamp_seconds`
- `ml_model_info{model_name,run_id}`
- `ml_model_feature_count`
- `ml_predict_response_bytes`

Alert rule yang diuji:
- High request rate: `sum(rate(ml_predict_requests_total[1m])) > 0.3`
- High latency p95: `histogram_quantile(0.95, sum(rate(ml_predict_latency_seconds_bucket[1m])) by (le)) > 0.75`
- Prediction errors: `sum(increase(ml_predict_errors_total[1m])) > 1`

Notifikasi dikirim ke webhook `http://127.0.0.1:8099/webhook` dan dicatat di `8.webhook_notifications.jsonl`.

Cara menjalankan:

```bash
cd "Monitoring dan Logging"

# 1. Jalankan model serving (dari folder Membangun_model, dengan run ID di atas)
mlflow models serve -m runs:/94e9087c03c24c5c9eeaa6e98f0758ba/model -p 5001 --host 127.0.0.1 --env-manager local

# 2. Jalankan exporter
python 3.prometheus_exporter.py

# 3. Jalankan Prometheus
prometheus --config.file=2.prometheus.yml

# 4. Jalankan Grafana dengan provisioning pada folder ini

# 5. Kirim trafik inferensi
python 7.inference.py --hold-seconds 600
```

Bukti pengujian (screenshot serving, Prometheus, Grafana, dan alerting) tersedia di folder `Monitoring dan Logging/`.
