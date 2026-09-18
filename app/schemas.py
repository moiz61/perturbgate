from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Telemetry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    voltage_jitter_mv: float = Field(ge=0.0, le=120.0)
    packet_retransmit_pct: float = Field(ge=0.0, le=30.0)
    inference_latency_ms: float = Field(ge=5.0, le=500.0)
    sensor_drift_sigma: float = Field(ge=0.0, le=8.0)
    cpu_temp_c: float = Field(ge=25.0, le=105.0)


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    telemetry: Telemetry
    stability_threshold: float = Field(default=0.90, ge=0.0, le=1.0)
    max_probability_shift: float = Field(default=0.18, ge=0.0, le=1.0)


class Stability(BaseModel):
    agreement: float
    max_probability_shift: float
    perturbations_evaluated: int
    fragile: bool


class PredictionResponse(BaseModel):
    label: Literal["healthy", "unstable"]
    probability_unstable: float
    decision: Literal["accept", "abstain"]
    stability: Stability
    model_version: str


class BatchPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[PredictionRequest] = Field(
        min_length=1,
        max_length=64,
    )


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    count: int
    model_version: str
