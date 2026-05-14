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
    runtime_targets: tuple[str, ...] = ()
    classes: tuple[str, ...] = ()
    license_name: str = ""
    description: str = ""
    family: str = ""
    size_tier: str = ""
    prompt_support: bool = False


ARTIFACT_MANIFESTS: dict[str, ArtifactSpec] = {
    "yolox-nano": ArtifactSpec(
        model_key="yolox-nano",
        stage="detector",
        upstream_url="https://github.com/Megvii-BaseDetection/YOLOX",
        sha256="",
        cache_filename="yolox_nano.pth",
        backends=("torch", "onnx"),
        runtime_targets=("cpu",),
        classes=("person", "backpack", "handbag", "suitcase"),
        license_name="Apache-2.0",
        description="Small-footprint commodity detector for PERSON and BAG mapping.",
        family="yolox",
        size_tier="small",
    ),
    "yolox-tiny": ArtifactSpec(
        model_key="yolox-tiny",
        stage="detector",
        upstream_url="https://github.com/Megvii-BaseDetection/YOLOX",
        sha256="",
        cache_filename="yolox_tiny.pth",
        backends=("torch", "onnx"),
        runtime_targets=("cpu", "cuda"),
        classes=("person", "backpack", "handbag", "suitcase"),
        license_name="Apache-2.0",
        description="Balanced YOLOX detector for CPU and edge compatibility.",
        family="yolox",
        size_tier="small",
    ),
    "yolox-s": ArtifactSpec(
        model_key="yolox-s",
        stage="detector",
        upstream_url="https://github.com/Megvii-BaseDetection/YOLOX",
        sha256="",
        cache_filename="yolox_s.pth",
        backends=("torch", "onnx"),
        runtime_targets=("cpu", "cuda"),
        classes=("person", "backpack", "handbag", "suitcase"),
        license_name="Apache-2.0",
        description="Commodity detector surface for PERSON and BAG mapping.",
        family="yolox",
        size_tier="small",
    ),
    "yolox-m": ArtifactSpec(
        model_key="yolox-m",
        stage="detector",
        upstream_url="https://github.com/Megvii-BaseDetection/YOLOX",
        sha256="",
        cache_filename="yolox_m.pth",
        backends=("torch", "onnx"),
        runtime_targets=("cuda",),
        classes=("person", "backpack", "handbag", "suitcase"),
        license_name="Apache-2.0",
        description="Higher-capacity YOLOX detector for GPU-oriented deployments.",
        family="yolox",
        size_tier="medium",
    ),
    "bytetrack": ArtifactSpec(
        model_key="bytetrack",
        stage="tracker",
        upstream_url="https://github.com/FoundationVision/ByteTrack",
        sha256="",
        cache_filename="bytetrack_x_mot17.pth.tar",
        backends=("python",),
        runtime_targets=("cpu", "cuda", "mlx"),
        license_name="MIT",
        description="Tracker compatibility adapter with ByteTrack-like assignment semantics.",
        family="bytetrack",
        size_tier="small",
    ),
    "bytetrack-s": ArtifactSpec(
        model_key="bytetrack-s",
        stage="tracker",
        upstream_url="https://github.com/FoundationVision/ByteTrack",
        sha256="",
        cache_filename="bytetrack_s_mot17.pth.tar",
        backends=("python",),
        runtime_targets=("cpu", "cuda", "mlx"),
        license_name="MIT",
        description="Smaller ByteTrack variant for registry/catalog selection.",
        family="bytetrack",
        size_tier="small",
    ),
    "bytetrack-balanced": ArtifactSpec(
        model_key="bytetrack-balanced",
        stage="tracker",
        upstream_url="https://github.com/FoundationVision/ByteTrack",
        sha256="",
        cache_filename="bytetrack_balanced.json",
        backends=("python",),
        runtime_targets=("cpu", "cuda", "mlx"),
        description="Balanced ByteTrack-style tracker profile for general surveillance scenes.",
        family="bytetrack",
        size_tier="small",
    ),
    "botsort": ArtifactSpec(
        model_key="botsort",
        stage="tracker",
        upstream_url="https://github.com/NirAharon/BoT-SORT",
        sha256="",
        cache_filename="botsort.json",
        backends=("python",),
        runtime_targets=("cpu", "cuda"),
        description="Commodity BoTSORT-style tracker profile with light feature awareness.",
        family="botsort",
        size_tier="small",
    ),
    "ocsort": ArtifactSpec(
        model_key="ocsort",
        stage="tracker",
        upstream_url="https://github.com/noahcao/OC_SORT",
        sha256="",
        cache_filename="ocsort.json",
        backends=("python",),
        runtime_targets=("cpu", "cuda", "mlx"),
        description="OCSORT-style tracker profile optimized for short occlusions.",
        family="ocsort",
        size_tier="small",
    ),
    "strongsort": ArtifactSpec(
        model_key="strongsort",
        stage="tracker",
        upstream_url="https://github.com/dyhBUPT/StrongSORT",
        sha256="",
        cache_filename="strongsort.json",
        backends=("python",),
        runtime_targets=("cpu", "cuda"),
        description="StrongSORT-style tracker profile with stronger feature matching.",
        family="strongsort",
        size_tier="medium",
    ),
    "cmtrack": ArtifactSpec(
        model_key="cmtrack",
        stage="tracker",
        upstream_url="https://github.com/FoundationVision/ByteTrack",
        sha256="",
        cache_filename="cmtrack.json",
        backends=("python",),
        runtime_targets=("cpu", "cuda", "mlx"),
        description="CMTrack compatibility profile for legacy Hearthlight tracker configurations.",
        family="cmtrack",
        size_tier="small",
    ),
    "transreid-market1501": ArtifactSpec(
        model_key="transreid-market1501",
        stage="reid",
        upstream_url="https://github.com/damo-cv/TransReID",
        sha256="",
        cache_filename="transreid_market1501.pth",
        backends=("torch", "embedding"),
        runtime_targets=("cpu", "cuda"),
        license_name="MIT",
        description="Person ReID and feature extraction baseline.",
        family="transreid",
        size_tier="small",
    ),
    "transreid-msmt17": ArtifactSpec(
        model_key="transreid-msmt17",
        stage="reid",
        upstream_url="https://github.com/damo-cv/TransReID",
        sha256="",
        cache_filename="transreid_msmt17.pth",
        backends=("torch", "embedding"),
        runtime_targets=("cuda",),
        license_name="MIT",
        description="Higher-diversity TransReID manifest for larger person galleries.",
        family="transreid",
        size_tier="medium",
    ),
    "rtmo-s": ArtifactSpec(
        model_key="rtmo-s",
        stage="pose",
        upstream_url="https://github.com/open-mmlab/mmpose",
        sha256="",
        cache_filename="rtmo_s.onnx",
        backends=("onnx",),
        runtime_targets=("cpu", "cuda"),
        license_name="Apache-2.0",
        description="Pose estimation compatibility surface for person keypoint attachment.",
        family="rtmo",
        size_tier="small",
    ),
    "rtmo-m": ArtifactSpec(
        model_key="rtmo-m",
        stage="pose",
        upstream_url="https://github.com/open-mmlab/mmpose",
        sha256="",
        cache_filename="rtmo_m.onnx",
        backends=("onnx",),
        runtime_targets=("cuda",),
        license_name="Apache-2.0",
        description="Larger pose-estimation manifest for improved person keypoint fidelity.",
        family="rtmo",
        size_tier="medium",
    ),
    "heuristic-presence": ArtifactSpec(
        model_key="heuristic-presence",
        stage="anomaly_stage_1",
        upstream_url="https://github.com/lauretta-io/hearthlight_model_zoo",
        sha256="",
        cache_filename="heuristic_presence.json",
        backends=("python",),
        runtime_targets=("cpu", "cuda", "mlx"),
        license_name="Apache-2.0",
        description="Heuristic presence anomaly scoring fallback.",
        family="heuristic",
        size_tier="small",
    ),
    "siglip-stage-1-cpu": ArtifactSpec(
        model_key="siglip-stage-1-cpu",
        stage="anomaly_stage_1",
        upstream_url="https://huggingface.co/google/siglip-base-patch16-224",
        sha256="",
        cache_filename="siglip_stage_1_cpu.safetensors",
        backends=("torch",),
        runtime_targets=("cpu",),
        classes=("person", "backpack", "handbag", "suitcase"),
        license_name="Apache-2.0",
        description="Lightweight SigLIP stage-1 scorer tuned for CPU fallback deployments.",
        family="siglip",
        size_tier="small",
        prompt_support=True,
    ),
    "siglip-stage-1-cuda": ArtifactSpec(
        model_key="siglip-stage-1-cuda",
        stage="anomaly_stage_1",
        upstream_url="https://huggingface.co/google/siglip-base-patch16-224",
        sha256="",
        cache_filename="siglip_stage_1_cuda.safetensors",
        backends=("torch",),
        runtime_targets=("cuda",),
        classes=("person", "backpack", "handbag", "suitcase"),
        license_name="Apache-2.0",
        description="SigLIP stage-1 scorer for CUDA-backed anomaly prescreening.",
        family="siglip",
        size_tier="small",
        prompt_support=True,
    ),
    "siglip-stage-1-mlx": ArtifactSpec(
        model_key="siglip-stage-1-mlx",
        stage="anomaly_stage_1",
        upstream_url="https://huggingface.co/google/siglip-base-patch16-224",
        sha256="",
        cache_filename="siglip_stage_1_mlx.safetensors",
        backends=("mlx",),
        runtime_targets=("mlx",),
        classes=("person", "backpack", "handbag", "suitcase"),
        license_name="Apache-2.0",
        description="SigLIP stage-1 scorer packaged for Apple Silicon MLX runtimes.",
        family="siglip",
        size_tier="small",
        prompt_support=True,
    ),
    "smolvlm-stage-2-cpu": ArtifactSpec(
        model_key="smolvlm-stage-2-cpu",
        stage="anomaly_stage_2",
        upstream_url="https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        sha256="",
        cache_filename="smolvlm_stage_2_cpu.gguf",
        backends=("torch",),
        runtime_targets=("cpu",),
        license_name="Apache-2.0",
        description="SmolVLM stage-2 prompt interpreter for CPU-first narrative anomaly labeling.",
        family="smolvlm",
        size_tier="small",
        prompt_support=True,
    ),
    "smolvlm-stage-2-cuda": ArtifactSpec(
        model_key="smolvlm-stage-2-cuda",
        stage="anomaly_stage_2",
        upstream_url="https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        sha256="",
        cache_filename="smolvlm_stage_2_cuda.gguf",
        backends=("torch",),
        runtime_targets=("cuda",),
        license_name="Apache-2.0",
        description="SmolVLM stage-2 prompt interpreter accelerated for CUDA deployments.",
        family="smolvlm",
        size_tier="small",
        prompt_support=True,
    ),
    "smolvlm-stage-2-mlx": ArtifactSpec(
        model_key="smolvlm-stage-2-mlx",
        stage="anomaly_stage_2",
        upstream_url="https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        sha256="",
        cache_filename="smolvlm_stage_2_mlx.gguf",
        backends=("mlx",),
        runtime_targets=("mlx",),
        license_name="Apache-2.0",
        description="MLX-converted SmolVLM stage-2 interpreter for Apple Silicon anomaly reasoning.",
        family="smolvlm",
        size_tier="small",
        prompt_support=True,
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
