from __future__ import annotations

import asyncio
import json
from typing import Any

import numpy as np


class AnomalyDetector:
    def __init__(self, cfg: Any):
        self.threshold = float(cfg.get("threshold", 0.5))

    def __call__(self, frames) -> tuple[float, bool]:
        if not frames:
            return 0.0, False
        brightness_values = [
            float(np.asarray(frame).mean()) / 255.0
            for frame in frames
        ]
        score = max(brightness_values) if brightness_values else 0.0
        return score, score >= self.threshold


class VLLMAgent:
    def __init__(self, _cfg: Any):
        pass

    async def __call__(self, frames, *, extra_messages=None, extra_body=None):
        del frames, extra_messages, extra_body
        return json.dumps(
            {
                "scene_summary": "Heuristic anomaly adapter fallback",
                "confidence": 0.0,
                "anomaly_category": "",
            }
        )


class AnomalyDescriber:
    def __init__(
        self,
        cfg: Any,
        *,
        extra_messages: list[dict[str, object]] | None = None,
        extra_body: dict[str, object] | None = None,
    ):
        self.anomaly_detector = AnomalyDetector(getattr(cfg, "anomaly", cfg))
        self.vllm_agent = VLLMAgent(getattr(cfg, "vllm", {}))
        self.extra_messages = extra_messages or []
        self.extra_body = extra_body or {}

    def __call__(self, frames):
        result = asyncio.run(
            score_and_describe(
                frames,
                detector=self.anomaly_detector,
                agent=self.vllm_agent,
                extra_messages=self.extra_messages,
                extra_body=self.extra_body,
            )
        )
        return (
            bool(result["anomaly_detected"]),
            float(result["confidence"]),
            result["scene_summary"],
            str(result["anomaly_category"]),
        )


async def score_and_describe(
    frames,
    *,
    detector: AnomalyDetector,
    agent: VLLMAgent,
    extra_messages=None,
    extra_body=None,
) -> dict[str, object]:
    score, anomaly = detector(frames)
    if not anomaly:
        return {
            "anomaly_detected": False,
            "confidence": score,
            "scene_summary": None,
            "anomaly_category": "",
        }

    payload = await agent(
        frames,
        extra_messages=extra_messages,
        extra_body=extra_body,
    )
    decoded = json.loads(payload)
    confidence = decoded.get("confidence", score)
    return {
        "anomaly_detected": bool(confidence is not None and confidence > 0.5),
        "confidence": confidence,
        "scene_summary": decoded.get("scene_summary"),
        "anomaly_category": decoded.get("anomaly_category", ""),
    }
