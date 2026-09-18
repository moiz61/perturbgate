from __future__ import annotations

from pathlib import Path

import joblib

from app.model import calculate_sha256, load_model_bundle
from train import compute_model_version


MODEL_PATH = Path("artifacts/model.joblib")


def test_bundle_contains_semantic_model_version():
    bundle = joblib.load(MODEL_PATH)

    assert "model_version" in bundle
    assert len(bundle["model_version"]) == 64


def test_loaded_model_uses_semantic_version():
    bundle = joblib.load(MODEL_PATH)
    loaded = load_model_bundle(MODEL_PATH)

    assert loaded.model_version == bundle["model_version"]


def test_artifact_sha_is_tracked_separately():
    loaded = load_model_bundle(MODEL_PATH)

    assert (
        loaded.artifact_sha256
        == calculate_sha256(MODEL_PATH)
    )

    assert len(loaded.artifact_sha256) == 64


def test_semantic_version_can_be_recomputed():
    bundle = joblib.load(MODEL_PATH)

    recomputed = compute_model_version(
        bundle["model"]
    )

    assert recomputed == bundle["model_version"]
