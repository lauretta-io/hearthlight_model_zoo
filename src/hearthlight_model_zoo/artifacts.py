from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
from typing import Iterable
from urllib import request


@dataclass(frozen=True)
class ArtifactSpec:
    model_key: str
    stage: str
    upstream_url: str
    sha256: str
    cache_filename: str
    backends: tuple[str, ...]
    classes: tuple[str, ...] = ()
    license_name: str = ""
    description: str = ""
    family: str = ""


ARTIFACT_MANIFESTS: dict[str, ArtifactSpec] = {
    "yolox-nano": ArtifactSpec(
        model_key="yolox-nano",
        stage="detector",
        upstream_url="https://github.com/Megvii-BaseDetection/YOLOX",
        sha256="",
        cache_filename="yolox_nano.pth",
        backends=("torch", "onnx"),
        classes=("person", "backpack", "handbag", "suitcase"),
        license_name="Apache-2.0",
        description="Small-footprint commodity detector for PERSON and BAG mapping.",
        family="yolox",
    ),
    "yolox-tiny": ArtifactSpec(
        model_key="yolox-tiny",
        stage="detector",
        upstream_url="https://github.com/Megvii-BaseDetection/YOLOX",
        sha256="",
        cache_filename="yolox_tiny.pth",
        backends=("torch", "onnx"),
        classes=("person", "backpack", "handbag", "suitcase"),
        license_name="Apache-2.0",
        description="Balanced YOLOX detector for CPU and edge compatibility.",
        family="yolox",
    ),
    "yolox-s": ArtifactSpec(
        model_key="yolox-s",
        stage="detector",
        upstream_url="https://github.com/Megvii-BaseDetection/YOLOX",
        sha256="",
        cache_filename="yolox_s.pth",
        backends=("torch", "onnx"),
        classes=("person", "backpack", "handbag", "suitcase"),
        license_name="Apache-2.0",
        description="Commodity detector surface for PERSON and BAG mapping.",
        family="yolox",
    ),
    "yolox-m": ArtifactSpec(
        model_key="yolox-m",
        stage="detector",
        upstream_url="https://github.com/Megvii-BaseDetection/YOLOX",
        sha256="",
        cache_filename="yolox_m.pth",
        backends=("torch", "onnx"),
        classes=("person", "backpack", "handbag", "suitcase"),
        license_name="Apache-2.0",
        description="Higher-capacity YOLOX detector for GPU-oriented deployments.",
        family="yolox",
    ),
    "bytetrack": ArtifactSpec(
        model_key="bytetrack",
        stage="tracker",
        upstream_url="https://github.com/FoundationVision/ByteTrack",
        sha256="",
        cache_filename="bytetrack_x_mot17.pth.tar",
        backends=("python",),
        license_name="MIT",
        description="Tracker compatibility adapter with ByteTrack-like assignment semantics.",
        family="bytetrack",
    ),
    "bytetrack-s": ArtifactSpec(
        model_key="bytetrack-s",
        stage="tracker",
        upstream_url="https://github.com/FoundationVision/ByteTrack",
        sha256="",
        cache_filename="bytetrack_s_mot17.pth.tar",
        backends=("python",),
        license_name="MIT",
        description="Smaller ByteTrack variant for registry/catalog selection.",
        family="bytetrack",
    ),
    "transreid-market1501": ArtifactSpec(
        model_key="transreid-market1501",
        stage="reid",
        upstream_url="https://github.com/damo-cv/TransReID",
        sha256="",
        cache_filename="transreid_market1501.pth",
        backends=("torch", "embedding"),
        license_name="MIT",
        description="Person ReID and feature extraction baseline.",
        family="transreid",
    ),
    "transreid-msmt17": ArtifactSpec(
        model_key="transreid-msmt17",
        stage="reid",
        upstream_url="https://github.com/damo-cv/TransReID",
        sha256="",
        cache_filename="transreid_msmt17.pth",
        backends=("torch", "embedding"),
        license_name="MIT",
        description="Higher-diversity TransReID manifest for larger person galleries.",
        family="transreid",
    ),
    "rtmo-s": ArtifactSpec(
        model_key="rtmo-s",
        stage="pose",
        upstream_url="https://github.com/open-mmlab/mmpose",
        sha256="",
        cache_filename="rtmo_s.onnx",
        backends=("onnx",),
        license_name="Apache-2.0",
        description="Pose estimation compatibility surface for person keypoint attachment.",
        family="rtmo",
    ),
    "rtmo-m": ArtifactSpec(
        model_key="rtmo-m",
        stage="pose",
        upstream_url="https://github.com/open-mmlab/mmpose",
        sha256="",
        cache_filename="rtmo_m.onnx",
        backends=("onnx",),
        license_name="Apache-2.0",
        description="Larger pose-estimation manifest for improved person keypoint fidelity.",
        family="rtmo",
    ),
    "heuristic-presence": ArtifactSpec(
        model_key="heuristic-presence",
        stage="anomaly_stage_1",
        upstream_url="https://github.com/lauretta-io/hearthlight_model_zoo",
        sha256="",
        cache_filename="heuristic_presence.json",
        backends=("python",),
        license_name="Apache-2.0",
        description="Heuristic presence anomaly scoring fallback.",
        family="heuristic",
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
