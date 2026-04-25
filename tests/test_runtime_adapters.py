from types import SimpleNamespace
import asyncio
import unittest

import numpy as np

from hearthlight_model_zoo.anomaly_detectors import (
    AnomalyDescriber,
    AnomalyDetector,
    VLLMAgent,
    score_and_describe,
)
from hearthlight_model_zoo.detectors import Detector
from hearthlight_model_zoo.feature_extractors import FeatureExtractor
from hearthlight_model_zoo.pose import PoseDetector
from hearthlight_model_zoo.reid import PersonReIDBundle
from hearthlight_model_zoo.trackers import get_tracker


class RuntimeAdapterTests(unittest.TestCase):
    def test_detector_returns_compatible_empty_results(self):
        detector = Detector("yolox-s")
        outputs = detector([np.zeros((8, 8, 3), dtype=np.uint8)])
        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0].shape, (0, 6))

    def test_feature_extractor_returns_embeddings(self):
        extractor = FeatureExtractor("transreid-market1501")
        outputs = extractor([np.zeros((8, 8, 3), dtype=np.uint8)])
        self.assertEqual(outputs.shape[0], 1)
        self.assertGreater(outputs.shape[1], 8)

    def test_tracker_assigns_track_ids(self):
        tracker = get_tracker("bytetrack")
        outputs = tracker.update(np.array([[0, 0, 10, 10, 0.9]], dtype=np.float32))
        self.assertEqual(outputs.shape, (1, 5))
        self.assertEqual(outputs[0, 4], 1)

    def test_tracker_variants_are_available(self):
        for tracker_name in ("botsort", "ocsort", "strongsort", "cmtrack", "bytetrack-balanced"):
            tracker = get_tracker(tracker_name)
            outputs = tracker.update(np.array([[0, 0, 10, 10, 0.9]], dtype=np.float32))
            self.assertEqual(outputs.shape, (1, 5))

    def test_reid_bundle_search_and_updates(self):
        cfg = SimpleNamespace(
            reid={
                "person": {"high_threshold": 0.6, "low_threshold": 0.5, "min_features": 1},
                "bag": {"distance_threshold": 50.0, "min_features": 1},
            }
        )
        bundle = PersonReIDBundle(cfg, registration={})
        track = SimpleNamespace(track_id=1, feature=np.ones(128, dtype=np.float32), bbox=[0, 0, 10, 10])
        bundle.person_reid.reid([track])
        ids = bundle.person_reid.entities.get_ids([track])
        self.assertTrue(ids[0] > 0)
        search = bundle.search(np.ones(128, dtype=np.float32), 5, 0.5)
        self.assertTrue(search)

    def test_pose_detector_passes_through_without_runtime(self):
        cfg = SimpleNamespace(pose={"device": "cpu", "model_name": "rtmo-s"})
        detector = PoseDetector(cfg)
        result = detector(SimpleNamespace(frames=[]), [])
        self.assertEqual(result, [])

    def test_anomaly_adapter_returns_structured_payload(self):
        detector = AnomalyDetector({"threshold": 0.1})
        agent = VLLMAgent({})
        result = asyncio.run(
            score_and_describe(
                [np.full((8, 8, 3), 255, dtype=np.uint8)],
                detector=detector,
                agent=agent,
            )
        )
        self.assertIn("confidence", result)
        self.assertIn("anomaly_detected", result)

    def test_anomaly_describer_wraps_detector_and_agent(self):
        cfg = SimpleNamespace(anomaly={"threshold": 0.1}, vllm={})
        describer = AnomalyDescriber(cfg)
        anomaly_detected, score, scene_summary, category = describer(
            [np.full((8, 8, 3), 255, dtype=np.uint8)]
        )
        self.assertFalse(anomaly_detected)
        self.assertGreaterEqual(score, 0.0)
        self.assertEqual(scene_summary, "Heuristic anomaly adapter fallback")
        self.assertEqual(category, "")


if __name__ == "__main__":
    unittest.main()
