# hearthlight_model_zoo

Commodity model adapters for Hearthlight.

This package exposes a stable import surface for:

- `hearthlight_model_zoo.detectors.Detector`
- `hearthlight_model_zoo.trackers.get_tracker`
- `hearthlight_model_zoo.feature_extractors.FeatureExtractor`
- `hearthlight_model_zoo.reid.PersonReIDBundle`

The package keeps only code plus artifact manifests. Checkpoints are expected to be
resolved into `~/.cache/hearthlight_model_zoo` at runtime.

