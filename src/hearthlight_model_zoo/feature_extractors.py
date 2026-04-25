from __future__ import annotations

import inspect

import numpy as np


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm <= 0:
        return vector.astype(np.float32)
    return (vector / norm).astype(np.float32)


def _resize_like(image: np.ndarray, *, height: int = 8, width: int = 8) -> np.ndarray:
    if image.size == 0:
        return np.zeros((height, width, 3), dtype=np.float32)
    src_height, src_width = image.shape[:2]
    if src_height == 0 or src_width == 0:
        return np.zeros((height, width, 3), dtype=np.float32)
    y_idx = np.linspace(0, src_height - 1, num=height).round().astype(int)
    x_idx = np.linspace(0, src_width - 1, num=width).round().astype(int)
    resized = image[np.ix_(y_idx, x_idx)]
    if resized.ndim == 2:
        resized = np.repeat(resized[..., None], 3, axis=2)
    return resized.astype(np.float32)


class FeatureExtractor:
    def __init__(self, model_name: str, *, backend: str = "numpy", precision: str = "fp32", device: str = "cpu"):
        self.model_name = model_name
        self.backend = backend
        self.precision = precision
        self.device = device
        self.signature = inspect.signature(self.__class__)

    def _extract_single(self, image: np.ndarray) -> np.ndarray:
        resized = _resize_like(image)
        channels = resized.mean(axis=(0, 1))
        flattened = resized.reshape(-1)
        histogram, _ = np.histogram(flattened, bins=125, range=(0, 255))
        vector = np.concatenate([channels, histogram.astype(np.float32)], axis=0)
        return _normalize_vector(vector)

    def __call__(self, images: list[np.ndarray]) -> np.ndarray:
        if not images:
            return np.empty((0, 128), dtype=np.float32)
        return np.vstack([self._extract_single(image) for image in images]).astype(np.float32)

