"""Exporter Prometheus yang membungkus MLflow model serving Telco Churn."""

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

MLFLOW_URL = os.environ.get("MLFLOW_URL", "http://127.0.0.1:5001/invocations")
EXPORTER_PORT = int(os.environ.get("EXPORTER_PORT", "8000"))
WEBHOOK_PORT = int(os.environ.get("WEBHOOK_PORT", "8099"))
WEBHOOK_LOG = os.environ.get(
    "WEBHOOK_LOG",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "webhook_notifications.jsonl"),
)

REQUEST_COUNT = Counter(
    "ml_predict_requests_total",
    "Total permintaan prediksi berdasarkan status respons",
    ["status"],
)
PREDICT_LATENCY = Histogram(
    "ml_predict_latency_seconds",
    "Durasi pemrosesan permintaan prediksi dalam detik",
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 2.0, 5.0),
)
PREDICTIONS = Counter(
    "ml_predictions_total",
    "Total hasil prediksi berdasarkan kelas",
    ["prediction"],
)
IN_PROGRESS = Gauge(
    "ml_predict_in_progress",
    "Jumlah permintaan prediksi yang sedang diproses",
)
BATCH_SIZE = Histogram(
    "ml_request_batch_size",
    "Jumlah baris pada setiap permintaan prediksi",
    buckets=(1, 2, 5, 10, 25, 50, 100),
)
ERRORS = Counter(
    "ml_predict_errors_total",
    "Total permintaan prediksi yang gagal berdasarkan jenis kesalahan",
    ["error_type"],
)
LAST_REQUEST = Gauge(
    "ml_last_request_timestamp_seconds",
    "Waktu Unix permintaan prediksi terakhir",
)
MODEL_INFO = Gauge(
    "ml_model_info",
    "Informasi model yang sedang dilayani",
    ["model_name", "run_id"],
)
FEATURE_COUNT = Gauge(
    "ml_model_feature_count",
    "Jumlah fitur yang dibutuhkan model",
)
RESPONSE_BYTES = Histogram(
    "ml_predict_response_bytes",
    "Ukuran respons prediksi dalam byte",
    buckets=(100, 500, 1000, 5000, 10000, 50000),
)

MODEL_INFO.labels(
    model_name="telco-churn-random-forest",
    run_id="94e9087c03c24c5c9eeaa6e98f0758ba",
).set(1)
FEATURE_COUNT.set(23)


def batch_length(payload):
    if "dataframe_split" in payload and "data" in payload["dataframe_split"]:
        return len(payload["dataframe_split"]["data"])
    if "dataframe_records" in payload:
        return len(payload["dataframe_records"])
    if "instances" in payload:
        return len(payload["instances"])
    return 0


def forward(payload, timeout=60):
    request = Request(
        MLFLOW_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        return response.status, response.read()


def count_predictions(body):
    try:
        predictions = json.loads(body).get("predictions", [])
    except (ValueError, AttributeError):
        return 0
    for prediction in predictions:
        PREDICTIONS.labels(prediction=str(prediction)).inc()
    return len(predictions)


class PredictHandler(BaseHTTPRequestHandler):
    server_version = "ChurnExporter/1.0"

    def log_message(self, fmt, *args):
        print(f"[exporter] {self.address_string()} {fmt % args}", flush=True)

    def send_json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/metrics":
            body = generate_latest()
            self.send_response(200)
            self.send_header("Content-Type", CONTENT_TYPE_LATEST)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/health":
            self.send_json(200, {"status": "ok", "in_progress": IN_PROGRESS._value.get()})
            return
        self.send_json(200, {"service": "telco-churn-exporter", "metrics": "/metrics"})

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path not in ("/predict", "/invocations"):
            self.send_json(404, {"error": "endpoint tidak ditemukan"})
            return

        started = time.perf_counter()
        IN_PROGRESS.inc()
        status = 500
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 10_000_000:
                REQUEST_COUNT.labels(status="400").inc()
                ERRORS.labels(error_type="validation").inc()
                LAST_REQUEST.set(time.time())
                self.send_json(400, {"error": "body JSON tidak valid"})
                status = 400
                return

            try:
                payload = json.loads(self.rfile.read(length))
            except ValueError:
                REQUEST_COUNT.labels(status="400").inc()
                ERRORS.labels(error_type="validation").inc()
                LAST_REQUEST.set(time.time())
                self.send_json(400, {"error": "body bukan JSON yang valid"})
                status = 400
                return

            if not isinstance(payload, dict) or not any(
                key in payload for key in ("dataframe_split", "dataframe_records", "instances")
            ):
                REQUEST_COUNT.labels(status="400").inc()
                ERRORS.labels(error_type="validation").inc()
                LAST_REQUEST.set(time.time())
                self.send_json(400, {"error": "format payload tidak dikenali"})
                status = 400
                return

            delay_ms = parse_qs(parsed.query).get("delay_ms", ["0"])[0]
            try:
                delay = max(0.0, float(delay_ms) / 1000.0)
            except ValueError:
                delay = 0.0
            if delay:
                time.sleep(delay)

            BATCH_SIZE.observe(batch_length(payload))

            try:
                upstream_status, body = forward(payload)
            except HTTPError as exc:
                body = exc.read()
                error_type = "validation" if exc.code == 400 else "upstream"
                ERRORS.labels(error_type=error_type).inc()
                REQUEST_COUNT.labels(status="502").inc()
                LAST_REQUEST.set(time.time())
                RESPONSE_BYTES.observe(len(body))
                self.send_response(502)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                status = 502
                return
            except (URLError, TimeoutError):
                ERRORS.labels(error_type="connection").inc()
                REQUEST_COUNT.labels(status="503").inc()
                LAST_REQUEST.set(time.time())
                self.send_json(503, {"error": "model serving tidak dapat dihubungi"})
                status = 503
                return

            count = count_predictions(body)
            REQUEST_COUNT.labels(status=str(upstream_status)).inc()
            LAST_REQUEST.set(time.time())
            RESPONSE_BYTES.observe(len(body))
            self.send_response(upstream_status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            status = upstream_status
            print(f"[exporter] 200 predicted={count} elapsed={time.perf_counter() - started:.3f}s", flush=True)
        finally:
            PREDICT_LATENCY.observe(time.perf_counter() - started)
            IN_PROGRESS.dec()
            if status not in (200, 400, 502, 503):
                REQUEST_COUNT.labels(status=str(status)).inc()


class WebhookHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        record = {"received_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "payload": raw.decode("utf-8", "replace")}
        with open(WEBHOOK_LOG, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(f"[webhook] notifikasi diterima ({len(raw)} byte)", flush=True)
        body = b'{"status":"received"}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def start_webhook_server():
    server = ThreadingHTTPServer(("0.0.0.0", WEBHOOK_PORT), WebhookHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"[webhook] mendengarkan di http://127.0.0.1:{WEBHOOK_PORT}/webhook", flush=True)


def main():
    start_webhook_server()
    server = ThreadingHTTPServer(("0.0.0.0", EXPORTER_PORT), PredictHandler)
    print(f"[exporter] metrics di http://127.0.0.1:{EXPORTER_PORT}/metrics", flush=True)
    print(f"[exporter] prediksi di http://127.0.0.1:{EXPORTER_PORT}/predict", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
