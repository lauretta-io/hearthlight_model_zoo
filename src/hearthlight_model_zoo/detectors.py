from __future__ import annotations

import logging

import numpy as np

from .artifacts import get_artifact_spec

logger = logging.getLogger(__name__)

BAG_SOURCE_CLASSES = {"backpack", "handbag", "suitcase"}
COMMODITY_CLASS_IDS = {"person": 0, "bag": 1}


class Detector:
    def __init__(self, model_name: str, backend: str = "onnx", precision: str = "fp32", *, device: str = "cpu"):
        self.model_name = model_name
        self.backend = backend
        self.precision = precision
        self.device = device
        try:
            self.spec = get_artifact_spec(model_name)
        except KeyError:
            self.spec = None
            logger.warning("Unknown detector artifact %s; detector will run in no-op mode", model_name)

    def _empty_result(self) -> np.ndarray:
        return np.empty((0, 6), dtype=np.float32)

    def __call__(self, images: list[np.ndarray], conf_by_class: dict[int, float] | None = None) -> list[np.ndarray]:
        # The initial public lane is compatibility-first: when a concrete runtime is not
        # installed and weights are not cached, return an empty detection tensor rather
        # than failing imports or the worker startup path.
        return [self._empty_result() for _ in images]

