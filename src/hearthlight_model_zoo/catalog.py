from __future__ import annotations

from collections import defaultdict
import json
from importlib import resources
from pathlib import Path

from .artifacts import ARTIFACT_MANIFESTS, ArtifactSpec

MASTER_CATALOG_RESOURCE = "master_catalog.json"


def list_supported_stages() -> list[str]:
    return sorted({spec.stage for spec in ARTIFACT_MANIFESTS.values()})


def list_stage_models(stage: str) -> list[ArtifactSpec]:
    normalized = stage.strip().lower()
    return sorted(
        (spec for spec in ARTIFACT_MANIFESTS.values() if spec.stage == normalized),
        key=lambda spec: spec.model_key,
    )


def build_stage_catalog() -> dict[str, list[dict[str, object]]]:
    catalog = defaultdict(list)
    for spec in ARTIFACT_MANIFESTS.values():
        catalog[spec.stage].append(
            {
                "model_key": spec.model_key,
                "family": spec.family,
                "stage": spec.stage,
                "backends": list(spec.backends),
                "runtime_targets": list(spec.runtime_targets),
                "classes": list(spec.classes),
                "license": spec.license_name,
                "description": spec.description,
                "upstream_url": spec.upstream_url,
                "cache_filename": spec.cache_filename,
                "size_tier": spec.size_tier,
                "prompt_support": spec.prompt_support,
            }
        )
    return {
        stage: sorted(entries, key=lambda entry: str(entry["model_key"]))
        for stage, entries in catalog.items()
    }


def list_model_keys(stage: str | None = None) -> list[str]:
    if stage is None:
        return sorted(ARTIFACT_MANIFESTS)
    return [spec.model_key for spec in list_stage_models(stage)]


def load_master_catalog(path: str | Path | None = None) -> dict[str, object]:
    if path is not None:
        raw = Path(path).read_text()
    else:
        raw = resources.files("hearthlight_model_zoo").joinpath(MASTER_CATALOG_RESOURCE).read_text()
    return json.loads(raw)


def write_master_catalog(path: str | Path, catalog: dict[str, object]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(catalog, indent=2, sort_keys=True) + "\n")
    return target
