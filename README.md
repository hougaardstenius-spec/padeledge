# Padeledge

AI-assisted padel shot analysis with Streamlit.

## Golden Path (local)

### 1) Setup

```bash
python3.10 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
```

### 2) Train model

```bash
.venv/bin/python scripts/train_shot_model.py -v
```

Expected artifacts:
- `models/shot_classifier.pkl`
- `models/metrics.json`

### 3) Run app

```bash
.venv/bin/streamlit run streamlit_app.py
```

Open `http://localhost:8501` and use:
- `Match Analyzer` for inference on uploaded videos
- `Training Dashboard` for dataset/model overview

## Tests

Run smoke tests:

```bash
.venv/bin/pytest -q
```

The test suite validates:
- training script produces model + metrics
- `ShotDetector.analyze()` returns a valid output structure

## Training Pipeline

Only one training entrypoint is supported:

- `scripts/train_shot_model.py`

Legacy scripts under `training/` are archived and not used by the app.
