from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split


TRAINING_SEED = 42
DATASET_VERSION = "synthetic-risk-v1"

FEATURE_NAMES = [
    "voltage_jitter_mv",
    "packet_retransmit_pct",
    "inference_latency_ms",
    "sensor_drift_sigma",
    "cpu_temp_c",
]

MODEL_CONFIG = {
    "learning_rate": 0.08,
    "max_iter": 180,
    "max_leaf_nodes": 23,
    "l2_regularization": 0.4,
    "random_state": TRAINING_SEED,
}

MODEL_PROBES = np.array(
    [
        [0.0, 0.0, 5.0, 0.0, 25.0],
        [40.0, 5.0, 100.0, 2.0, 60.0],
        [65.0, 10.0, 160.0, 3.5, 65.0],
        [
            41.93188438245095,
            7.8001614283472795,
            403.65791719276825,
            3.380880401569116,
            102.42304400705142,
        ],
        [80.0, 15.0, 250.0, 4.0, 75.0],
        [120.0, 30.0, 500.0, 8.0, 105.0],
    ],
    dtype=np.float64,
)


def build_dataset(
    n: int = 8000,
    seed: int = TRAINING_SEED,
):
    rng = np.random.default_rng(seed)

    jitter = rng.uniform(0.0, 120.0, n)
    retransmit = rng.uniform(0.0, 30.0, n)
    latency = rng.uniform(5.0, 500.0, n)
    drift = rng.uniform(0.0, 8.0, n)
    temp = rng.uniform(25.0, 105.0, n)

    risk_logit = (
        -7.6
        + 0.040 * jitter
        + 0.115 * retransmit
        + 0.0045 * latency
        + 0.48 * drift
        + 0.025 * (temp - 55.0)
        + 0.0014 * jitter * retransmit
        + 0.22 * np.maximum(drift - 4.0, 0.0) ** 2
    )

    probability = 1.0 / (1.0 + np.exp(-risk_logit))
    y = rng.binomial(1, probability)

    X = np.column_stack(
        [
            jitter,
            retransmit,
            latency,
            drift,
            temp,
        ]
    )

    return X, y


def compute_model_version(
    model: HistGradientBoostingClassifier,
) -> str:
    probe_probabilities = model.predict_proba(
        MODEL_PROBES
    )[:, 1]

    fingerprint = {
        "dataset_version": DATASET_VERSION,
        "feature_names": FEATURE_NAMES,
        "training_seed": TRAINING_SEED,
        "model_class": type(model).__name__,
        "model_config": MODEL_CONFIG,
        "sklearn_version": sklearn.__version__,
        "probe_probabilities": [
            round(float(value), 12)
            for value in probe_probabilities
        ],
    }

    canonical = json.dumps(
        fingerprint,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(canonical).hexdigest()


def train(output: Path) -> None:
    X, y = build_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=TRAINING_SEED,
        stratify=y,
    )

    model = HistGradientBoostingClassifier(
        **MODEL_CONFIG
    )

    model.fit(X_train, y_train)

    probability = model.predict_proba(X_test)[:, 1]
    prediction = (probability >= 0.5).astype(int)

    metrics = {
        "accuracy": float(
            accuracy_score(y_test, prediction)
        ),
        "roc_auc": float(
            roc_auc_score(y_test, probability)
        ),
    }

    model_version = compute_model_version(model)

    bundle = {
        "model": model,
        "feature_names": FEATURE_NAMES,
        "metrics": metrics,
        "training_seed": TRAINING_SEED,
        "dataset_version": DATASET_VERSION,
        "model_version": model_version,
        "problem": (
            "synthetic edge-node instability classification"
        ),
    }

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        bundle,
        output,
        compress=3,
    )

    print(f"saved_model={output}")
    print(f"accuracy={metrics['accuracy']:.3f}")
    print(f"roc_auc={metrics['roc_auc']:.3f}")
    print(f"model_version={model_version}")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/model.joblib"),
    )

    args = parser.parse_args()

    train(args.output)


if __name__ == "__main__":
    main()
