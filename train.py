from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split


FEATURE_NAMES = [
    "voltage_jitter_mv",
    "packet_retransmit_pct",
    "inference_latency_ms",
    "sensor_drift_sigma",
    "cpu_temp_c",
]


def build_dataset(n: int = 8000, seed: int = 42):
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


def train(output: Path) -> None:
    X, y = build_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    model = HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_iter=180,
        max_leaf_nodes=23,
        l2_regularization=0.4,
        random_state=42,
    )

    model.fit(X_train, y_train)

    probability = model.predict_proba(X_test)[:, 1]
    prediction = (probability >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, prediction)),
        "roc_auc": float(roc_auc_score(y_test, probability)),
    }

    bundle = {
        "model": model,
        "feature_names": FEATURE_NAMES,
        "metrics": metrics,
        "training_seed": 42,
        "problem": "synthetic edge-node instability classification",
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output, compress=3)

    print(f"saved_model={output}")
    print(f"accuracy={metrics['accuracy']:.3f}")
    print(f"roc_auc={metrics['roc_auc']:.3f}")


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
