from __future__ import annotations

import asyncio
from types import SimpleNamespace

from app import main as main_module
from app.schemas import (
    PredictionRequest,
    PredictionResponse,
    Stability,
    Telemetry,
)


def test_inference_semaphore_limits_concurrency_to_four(
    monkeypatch,
):
    async def scenario():
        active = 0
        maximum_active = 0

        async def fake_run_in_threadpool(
            _function,
            *_args,
        ):
            nonlocal active
            nonlocal maximum_active

            active += 1
            maximum_active = max(
                maximum_active,
                active,
            )

            # Keep the fake inference occupied long enough
            # for competing requests to reach the semaphore.
            await asyncio.sleep(0.05)

            active -= 1

            return PredictionResponse(
                label="healthy",
                probability_unstable=0.1,
                decision="accept",
                stability=Stability(
                    agreement=1.0,
                    max_probability_shift=0.01,
                    perturbations_evaluated=10,
                    fragile=False,
                ),
                model_version="test",
            )

        monkeypatch.setattr(
            main_module,
            "run_in_threadpool",
            fake_run_in_threadpool,
        )

        state = SimpleNamespace(
            ready=True,
            loaded_model=object(),
            model_version="test",
            inference_semaphore=asyncio.Semaphore(4),
        )

        fake_request = SimpleNamespace(
            app=SimpleNamespace(
                state=state,
            )
        )

        payload = PredictionRequest(
            telemetry=Telemetry(
                voltage_jitter_mv=40,
                packet_retransmit_pct=5,
                inference_latency_ms=100,
                sensor_drift_sigma=2,
                cpu_temp_c=60,
            )
        )

        results = await asyncio.gather(
            *[
                main_module.predict_one(
                    fake_request,
                    payload,
                )
                for _ in range(12)
            ]
        )

        assert len(results) == 12
        assert maximum_active == 4

    asyncio.run(scenario())
