from pathlib import Path
import tempfile
import unittest

from hearthlight_model_zoo.artifacts import ensure_artifact, get_artifact_path, get_artifact_spec
from hearthlight_model_zoo.catalog import (
    build_stage_catalog,
    list_model_keys,
    list_stage_models,
    list_supported_stages,
    load_master_catalog,
    write_master_catalog,
)


class CatalogTests(unittest.TestCase):
    def test_supported_stages_include_new_runtime_surfaces(self):
        stages = list_supported_stages()
        self.assertIn("pose", stages)
        self.assertIn("anomaly_stage_1", stages)
        self.assertIn("anomaly_stage_2", stages)

    def test_stage_models_return_specs(self):
        detectors = list_stage_models("detector")
        self.assertTrue(any(spec.model_key == "yolox-s" for spec in detectors))
        self.assertTrue(any(spec.model_key == "yolox-m" for spec in detectors))

    def test_build_stage_catalog_contains_metadata(self):
        catalog = build_stage_catalog()
        self.assertIn("detector", catalog)
        self.assertTrue(any(row["license"] for row in catalog["detector"]))
        self.assertTrue(any(row["family"] == "yolox" for row in catalog["detector"]))

    def test_list_model_keys_supports_stage_filter(self):
        self.assertIn("transreid-market1501", list_model_keys("reid"))
        self.assertNotIn("bytetrack", list_model_keys("reid"))

    def test_master_catalog_contains_tracker_registrations(self):
        catalog = load_master_catalog()
        trackers = catalog["models"]["tracker"]
        self.assertIn("builtin_botsort", trackers)
        self.assertIn("builtin_ocsort", trackers)
        self.assertIn("strongsort", catalog["stage_options"]["tracker"])

    def test_master_catalog_detector_registration_includes_raw_classes(self):
        catalog = load_master_catalog()
        detector = catalog["models"]["detector"]["builtin_yolox_s_cpu"]
        self.assertEqual(
            detector["capabilities"]["classes"],
            ["person", "backpack", "handbag", "suitcase"],
        )

    def test_master_catalog_contains_backend_default_anomaly_registrations(self):
        catalog = load_master_catalog()
        stage_1 = catalog["models"]["anomaly_stage_1"]
        stage_2 = catalog["models"]["anomaly_stage_2"]
        self.assertEqual(stage_1["siglip_stage_1_cpu"]["default_for_backend"], "cpu")
        self.assertEqual(stage_1["siglip_stage_1_cuda"]["default_for_backend"], "cuda")
        self.assertEqual(stage_1["siglip_stage_1_mlx"]["default_for_backend"], "mlx")
        self.assertEqual(stage_2["smolvlm_stage_2_cpu"]["default_for_backend"], "cpu")
        self.assertEqual(stage_2["smolvlm_stage_2_cuda"]["default_for_backend"], "cuda")
        self.assertEqual(stage_2["smolvlm_stage_2_mlx"]["default_for_backend"], "mlx")
        self.assertIn("prompt_rules_stage_2", stage_2)
        self.assertIn("passthrough_stage_2", stage_2)

    def test_build_stage_catalog_exposes_runtime_targets_and_prompt_support(self):
        catalog = build_stage_catalog()
        stage_2 = {row["model_key"]: row for row in catalog["anomaly_stage_2"]}
        self.assertEqual(stage_2["smolvlm-stage-2-mlx"]["runtime_targets"], ["mlx"])
        self.assertTrue(stage_2["smolvlm-stage-2-cpu"]["prompt_support"])

    def test_master_catalog_round_trips_to_disk(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "master_catalog.json"
            catalog = {"version": 99, "models": {}, "stage_options": {"tracker": ["bytetrack"]}}
            written = write_master_catalog(target, catalog)
            self.assertEqual(written, target)
            self.assertEqual(load_master_catalog(target), catalog)

    def test_ensure_artifact_without_download_returns_cache_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            from os import environ

            original = environ.get("HEARTHLIGHT_MODEL_ZOO_CACHE_DIR")
            environ["HEARTHLIGHT_MODEL_ZOO_CACHE_DIR"] = temp_dir
            try:
                path = ensure_artifact("yolox-s", download=False)
                self.assertEqual(path, Path(temp_dir) / "yolox_s.pth")
                self.assertEqual(get_artifact_path("yolox-s"), path)
                self.assertEqual(get_artifact_spec("yolox-s").stage, "detector")
            finally:
                if original is None:
                    environ.pop("HEARTHLIGHT_MODEL_ZOO_CACHE_DIR", None)
                else:
                    environ["HEARTHLIGHT_MODEL_ZOO_CACHE_DIR"] = original


if __name__ == "__main__":
    unittest.main()
