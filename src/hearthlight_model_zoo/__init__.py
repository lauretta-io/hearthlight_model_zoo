from .artifacts import ARTIFACT_MANIFESTS, ArtifactSpec, ensure_artifact
from .catalog import (
    build_stage_catalog,
    list_model_keys,
    list_stage_models,
    list_supported_stages,
    load_master_catalog,
    write_master_catalog,
)

__all__ = [
    "AnomalyDescriber",
    "AnomalyDetector",
    "ARTIFACT_MANIFESTS",
    "ArtifactSpec",
    "build_stage_catalog",
    "Detector",
    "ensure_artifact",
    "FeatureExtractor",
    "get_tracker",
    "list_model_keys",
    "list_stage_models",
    "list_supported_stages",
    "load_master_catalog",
    "PersonReIDBundle",
    "PoseDetector",
    "VLLMAgent",
    "write_master_catalog",
]


def __getattr__(name: str):
    if name in {"AnomalyDescriber", "AnomalyDetector", "VLLMAgent"}:
        from .anomaly_detectors import AnomalyDescriber, AnomalyDetector, VLLMAgent

        return {
            "AnomalyDescriber": AnomalyDescriber,
            "AnomalyDetector": AnomalyDetector,
            "VLLMAgent": VLLMAgent,
        }[name]
    if name == "Detector":
        from .detectors import Detector

        return Detector
    if name == "FeatureExtractor":
        from .feature_extractors import FeatureExtractor

        return FeatureExtractor
    if name == "get_tracker":
        from .trackers import get_tracker

        return get_tracker
    if name == "PersonReIDBundle":
        from .reid import PersonReIDBundle

        return PersonReIDBundle
    if name == "PoseDetector":
        from .pose import PoseDetector

        return PoseDetector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
