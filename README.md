# hearthlight_model_zoo

Commodity model adapters for Hearthlight.

This package exposes a stable import surface for:

- `hearthlight_model_zoo.detectors.Detector`
- `hearthlight_model_zoo.trackers.get_tracker`
- `hearthlight_model_zoo.feature_extractors.FeatureExtractor`
- `hearthlight_model_zoo.reid.PersonReIDBundle`
- `hearthlight_model_zoo.pose.PoseDetector`
- `hearthlight_model_zoo.anomaly_detectors.AnomalyDetector`
- `hearthlight_model_zoo.anomaly_detectors.VLLMAgent`

The package keeps only code plus artifact manifests. Checkpoints are expected to be
resolved into `~/.cache/hearthlight_model_zoo` at runtime.

Shared runtime registrations live in `src/hearthlight_model_zoo/master_catalog.json`.
That file is intended to be the editable source of truth for shared detector, tracker,
reid, and anomaly stage options. Downstream repos can scan it and merge in local-only
registrations or overrides.

Current public catalog:

- detector: `yolox-nano`, `yolox-tiny`, `yolox-s`, `yolox-m`
- tracker: `bytetrack`, `bytetrack-s`, `bytetrack-balanced`, `botsort`, `ocsort`, `strongsort`, `cmtrack`
- reid / feature extractor: `transreid-market1501`, `transreid-msmt17`
- pose: `rtmo-s`, `rtmo-m`
- anomaly stage 1 heuristic: `heuristic-presence`

The runtime surface remains compatibility-first:

- detector output is normalized to Hearthlight’s `PERSON` and `BAG` tasks
- `GUN` is intentionally unsupported in the commodity detector lane
- pose and anomaly modules degrade to pass-through / heuristic behavior when their
  optional runtime dependencies or cached artifacts are unavailable
