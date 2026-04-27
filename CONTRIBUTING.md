# Contributing To Hearthlight Model Zoo

Thanks for improving `hearthlight_model_zoo`.

This repo is intentionally small: it is the shared model-management layer for Hearthlight, not the full application runtime. Most contributions fall into one of four buckets:

- add or adjust a shared registration in `src/hearthlight_model_zoo/master_catalog.json`
- add or adjust artifact metadata in `src/hearthlight_model_zoo/artifacts.py`
- extend a runtime adapter in `src/hearthlight_model_zoo/`
- keep documentation and tests aligned with the public package surface

## Files You Will Usually Touch

- `src/hearthlight_model_zoo/master_catalog.json`
  Shared registrations and stage option lists shipped with the package.
- `src/hearthlight_model_zoo/artifacts.py`
  Artifact metadata such as model keys, cache filenames, upstream URLs, backends, and license info.
- `src/hearthlight_model_zoo/`
  Runtime adapters and catalog helpers.
- `README.md`
  Public-facing onboarding and import examples.
- `tests/`
  Public API and catalog coverage.

## When To Update The Master Catalog

Edit `master_catalog.json` when you are changing shared registrations that downstream repos should discover automatically.

Common cases:

- adding a new detector, tracker, reid, pose, or anomaly registration
- changing a shared runtime alias such as `tracker_name`
- updating the stage option lists that downstream launchers or registry scanners should show
- changing shared healthcheck, capability, or resource metadata
- adding detector class metadata that downstream apps should expose in model pickers or model-library views

If a downstream repo should be able to discover the option without maintaining its own duplicate entry, it belongs in the master catalog.

## When To Update `artifacts.py`

Edit `artifacts.py` when you are changing artifact metadata for a model key.

Common cases:

- new upstream model family or variant
- new cache filename
- backend support changes
- new class mappings
- updated upstream URL, license, or description

`artifacts.py` answers the question: “what is this model artifact?”

`master_catalog.json` answers the question: “how should downstream apps register and expose it?”

## When You Need Both

Most new model additions should update both files:

- add the model key and artifact metadata to `artifacts.py`
- add the shared registration and stage option entry to `master_catalog.json`

For detectors, keep the catalog registration honest about the raw class surface. If the artifact can emit classes beyond Hearthlight's normalized `PERSON` and `BAG` mapping, include those raw detector classes in `capabilities.classes` so downstream apps can display them.

If a new tracker also needs a new alias or compatibility behavior, update the matching runtime adapter in `src/hearthlight_model_zoo/trackers.py`.

If a new detector or pose model changes public examples or inventory lists, update `README.md` too.

## Expectations By Model Family

- detectors should keep Hearthlight task normalization clear, especially for `PERSON` and `BAG`
- trackers should expose stable names through `get_tracker(...)`
- reid and feature extractor additions should keep the public import surface unchanged unless there is a strong compatibility reason
- pose and anomaly helpers should continue to fail soft when optional dependencies are unavailable

Prefer additive changes over breaking renames.

## Keep Docs And Inventory In Sync

Before opening a PR, make sure these stay aligned:

- `README.md` model family lists
- `README.md` import examples
- `master_catalog.json` stage options and registrations
- `artifacts.py` model keys
- tests in `tests/`

If a new model key exists in one place but not the others, downstream consumers get confusing behavior fast.

## Lightweight Validation

Use a virtual environment and install the package in editable mode:

```bash
python -m pip install -e ".[dev]"
```

Run the package tests:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

If you changed docs, also verify:

- the README image renders correctly
- internal links resolve
- every import example matches a real public symbol

## Pull Request Checklist

- the README wordmark image renders correctly on GitHub
- import examples match real symbols in the package
- `master_catalog.json` keys are consistent with `artifacts.py`
- tests pass locally
- README and `CONTRIBUTING.md` still describe the same workflow
