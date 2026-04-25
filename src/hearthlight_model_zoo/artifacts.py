from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
from typing import Iterable
from urllib import request


@dataclass(frozen=True)
class ArtifactSpec:
    model_key: str
    upstream_url: str
    sha256: str
    cache_filename: str
    backends: tuple[str, ...]
    classes: tuple[str, ...] = ()


ARTIFACT_MANIFESTS: dict[str, ArtifactSpec] = {
    "yolox-s": ArtifactSpec(
        model_key="yolox-s",
        upstream_url="https://github.com/Megvii-BaseDetection/YOLOX",
        sha256="",
        cache_filename="yolox_s.pth",
        backends=("torch", "onnx"),
        classes=("person", "backpack", "handbag", "suitcase"),
    ),
    "bytetrack": ArtifactSpec(
        model_key="bytetrack",
        upstream_url="https://github.com/FoundationVision/ByteTrack",
        sha256="",
        cache_filename="bytetrack_x_mot17.pth.tar",
        backends=("python",),
    ),
    "transreid-market1501": ArtifactSpec(
        model_key="transreid-market1501",
        upstream_url="https://github.com/damo-cv/TransReID",
        sha256="",
        cache_filename="transreid_market1501.pth",
        backends=("torch", "embedding"),
    ),
}


def get_cache_root() -> Path:
    configured = os.environ.get("HEARTHLIGHT_MODEL_ZOO_CACHE_DIR", "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".cache" / "hearthlight_model_zoo"


def iter_manifest_keys() -> Iterable[str]:
    return ARTIFACT_MANIFESTS.keys()


def get_artifact_spec(model_key: str) -> ArtifactSpec:
    normalized = model_key.strip()
    if normalized not in ARTIFACT_MANIFESTS:
        raise KeyError(f"unknown artifact {model_key}")
    return ARTIFACT_MANIFESTS[normalized]


def get_artifact_path(model_key: str) -> Path:
    spec = get_artifact_spec(model_key)
    return get_cache_root() / spec.cache_filename


def ensure_artifact(model_key: str, *, download: bool = False) -> Path:
    path = get_artifact_path(model_key)
    if path.exists() or not download:
        return path

    spec = get_artifact_spec(model_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    request.urlretrieve(spec.upstream_url, path)
    return path

