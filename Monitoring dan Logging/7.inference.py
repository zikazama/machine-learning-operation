"""Mengirim permintaan inferensi ke exporter untuk pengujian monitoring dan alerting."""

import argparse
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "namadataset_preprocessing" / "telco_churn_preprocessing.csv"


def build_payload(df: pd.DataFrame, start: int, size: int) -> dict:
    batch = df.iloc[start : start + size].drop(columns=["Churn"])
    return {
        "dataframe_split": {
            "columns": batch.columns.tolist(),
            "data": batch.values.tolist(),
        }
    }


def send(url: str, payload: dict, params: dict | None = None) -> tuple[requests.Response, float]:
    started = time.perf_counter()
    response = requests.post(url, json=payload, params=params, timeout=60)
    return response, time.perf_counter() - started


def print_result(label: str, response: requests.Response, elapsed: float) -> None:
    detail = ""
    if response.status_code == 200:
        detail = f" predictions={len(response.json().get('predictions', []))}"
    print(f"{label:<22} status={response.status_code} elapsed={elapsed:.3f}s{detail}", flush=True)


def malformed_payloads() -> list[dict]:
    return [
        {},
        {"dataframe_split": {"columns": ["unknown"], "data": [[1]]}},
        {"instances": []},
        {"dataframe_records": [{"tidak_ada": 1}]},
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000/predict")
    parser.add_argument("--hold-seconds", type=int, default=240)
    args = parser.parse_args()

    df = pd.read_csv(DATA_PATH)
    statuses = Counter()

    print(f"Dataset: {df.shape[0]} baris, {df.shape[1] - 1} fitur", flush=True)

    for index in range(2):
        response, elapsed = send(args.url, build_payload(df, index * 5, 2))
        statuses[response.status_code] += 1
        print_result(f"warmup-{index + 1}", response, elapsed)

    for index in range(8):
        size = 1 + index
        response, elapsed = send(args.url, build_payload(df, index * 7, size))
        statuses[response.status_code] += 1
        print_result(f"normal-{index + 1} (batch {size})", response, elapsed)

    for index in range(6):
        response, elapsed = send(
            args.url, build_payload(df, index * 3, 4), params={"delay_ms": 1200}
        )
        statuses[response.status_code] += 1
        print_result(f"latency-{index + 1}", response, elapsed)

    for index, payload in enumerate(malformed_payloads()):
        response, elapsed = send(args.url, payload)
        statuses[response.status_code] += 1
        print_result(f"error-{index + 1}", response, elapsed)

    deadline = time.time() + args.hold_seconds
    iteration = 0
    errors = malformed_payloads()
    print(f"Traffic campuran selama {args.hold_seconds} detik...", flush=True)
    while time.time() < deadline:
        iteration += 1
        mode = iteration % 6
        if mode == 3:
            response, elapsed = send(
                args.url, build_payload(df, iteration % 500, 4), params={"delay_ms": 1200}
            )
            label = f"hold-{iteration} (latency)"
        elif mode == 4:
            response, elapsed = send(args.url, errors[iteration % len(errors)])
            label = f"hold-{iteration} (error)"
        else:
            response, elapsed = send(
                args.url, build_payload(df, iteration % 500, 1 + iteration % 9)
            )
            label = f"hold-{iteration} (normal)"
        statuses[response.status_code] += 1
        if iteration % 10 == 0:
            print_result(label, response, elapsed)
        time.sleep(2)

    print("Ringkasan status respons:", flush=True)
    for status, total in sorted(statuses.items()):
        print(f"  {status}: {total}", flush=True)


if __name__ == "__main__":
    main()
