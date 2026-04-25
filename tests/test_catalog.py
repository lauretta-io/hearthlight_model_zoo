from pathlib import Path
import tempfile
import unittest

from hearthlight_model_zoo.artifacts import ensure_artifact, get_artifact_path, get_artifact_spec
from hearthlight_model_zoo.catalog import (
    build_stage_catalog,
    list_model_keys,
    list_stage_models,
    list_supported_stages,
)


class CatalogTests(unittest.TestCase):
    def test_supported_stages_include_new_runtime_surfaces(self):
        stages = list_supported_stages()
        self.assertIn("pose", stages)
        self.assertIn("anomaly_stage_1", stages)

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
