# Monitoring dan Logging — Telco Churn (fauzi19)

Implementasi monitoring dan logging untuk model Random Forest Telco Customer Churn
yang dilayani dengan MLflow Model Serving.

## Komponen

| Berkas | Keterangan |
|---|---|
| `1.bukti_serving/1.bukti_serving.jpg` | Bukti model dilayani `mlflow models serve` pada port 5001 dan merespons prediksi |
| `2.prometheus.yml` | Konfigurasi Prometheus, scrape exporter setiap 5 detik |
| `3.prometheus_exporter.py` | Exporter yang membungkus MLflow serving dan mengekspos 10 metrik Prometheus |
| `4.bukti monitoring Prometheus/` | Empat grafik metrik di halaman Query Prometheus (request rate, latensi p95, prediksi per kelas, error rate) dan satu tampilan Table yang memuat nilai angka empat metrik mentah |
| `5.bukti monitoring Grafana/` | Dashboard `fauzi19` beserta satu tangkapan layar untuk setiap metrik yang sama |
| `6.bukti alerting Grafana/` | Tiga alert rule berstatus Firing dan bukti notifikasi webhook terkirim |
| `7.inference.py` | Pengirim trafik inferensi: normal, latensi buatan, dan payload tidak valid |
| `requirements.txt` | Dependensi Python untuk exporter dan inference |
| `namadataset_preprocessing/` | Data uji hasil preprocessing |
| `8.webhook_notifications.jsonl` | Log payload notifikasi yang diterima endpoint webhook Grafana |

## Metrik yang diekspos

1. `ml_predict_requests_total{status}`
2. `ml_predict_latency_seconds`
3. `ml_predictions_total{prediction}`
4. `ml_predict_in_progress`
5. `ml_request_batch_size`
6. `ml_predict_errors_total{error_type}`
7. `ml_last_request_timestamp_seconds`
8. `ml_model_info{model_name,run_id}`
9. `ml_model_feature_count`
10. `ml_predict_response_bytes`

## Pemetaan metrik, bukti, dan panel

| Metrik | Bukti Prometheus | Panel Grafana |
|---|---|---|
| `ml_predict_requests_total` | `1.monitoring_request_rate.jpg` | `1.monitoring_request_rate.jpg`, `4.monitoring_requests_by_status.jpg` |
| `ml_predict_latency_seconds` | `2.monitoring_latency_p95.jpg` | `3.monitoring_latency_p95.jpg` |
| `ml_predictions_total` | `3.monitoring_predictions.jpg` | `2.monitoring_predictions.jpg` |
| `ml_predict_errors_total` | `4.monitoring_errors.jpg` | `5.monitoring_errors.jpg` |
| `ml_predict_in_progress` | — | `7.monitoring_in_progress.jpg` |
| `ml_request_batch_size` | — | `6.monitoring_batch_size.jpg` |
| `ml_last_request_timestamp_seconds` | — | `8.monitoring_last_request_age.jpg` |
| `ml_model_info`, `ml_model_feature_count` | — | `10.monitoring_model_info.jpg` |
| `ml_predict_response_bytes` | — | `9.monitoring_response_size.jpg` |

Setiap bukti Prometheus diambil dari halaman Query dan menampilkan ekspresi metrik beserta grafiknya, bukan halaman target health. Bukti `5.monitoring_tabel_nilai_metrik.jpg` memakai tab Table dan menampilkan nilai angka metrik mentah `ml_predict_requests_total`, `ml_predictions_total`, `ml_predict_errors_total`, dan `ml_predict_latency_seconds_count`. Seluruh metrik itu juga tampil pada dashboard Grafana bernama `fauzi19`, baik pada `dashboard_fauzi19.jpg` maupun pada tangkapan layar panel masing-masing.

## Alert yang diuji

- High request rate: `sum(rate(ml_predict_requests_total[1m])) > 0.3`
- High latency p95: `histogram_quantile(0.95, sum(rate(ml_predict_latency_seconds_bucket[1m])) by (le)) > 0.75`
- Prediction errors: `sum(increase(ml_predict_errors_total[1m])) > 1`

Notifikasi dikirim ke contact point webhook `k4-webhook`
(`http://127.0.0.1:8099/webhook`) dan dicatat pada `8.webhook_notifications.jsonl`.

## Cara menjalankan

```bash
# 1. Jalankan model serving (dari folder Membangun_model)
mlflow models serve -m runs:/94e9087c03c24c5c9eeaa6e98f0758ba/model \
  -p 5001 --host 127.0.0.1 --env-manager local

# 2. Jalankan exporter
python 3.prometheus_exporter.py

# 3. Jalankan Prometheus
prometheus --config.file=2.prometheus.yml

# 4. Jalankan Grafana dengan provisioning pada folder ini

# 5. Kirim trafik inferensi
python 7.inference.py --hold-seconds 600
```
