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
- `models/release_report.json`

Release gate thresholds can be configured with env vars:

- `PADELEDGE_MIN_ACCURACY` (default `0.65`)
- `PADELEDGE_MIN_MACRO_F1` (default `0.55`)

### 3) Run app

```bash
.venv/bin/streamlit run streamlit_app.py
```

Open `http://localhost:8501` and use:
- `Match Analyzer` for inference on uploaded videos
- `Training Dashboard` for dataset/model overview

Every upload analysis is logged to:

- `data/analysis_logs/match_analyses.jsonl`

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

## Label Protocol Files

Use the template:

- `data/labels_template.csv`

Generate a draft labels file from clips under `data/samples`:

```bash
.venv/bin/python scripts/bootstrap_labels.py --samples-root data/samples --output data/labels.csv --default-match-id match_001 --default-player-id player_001
```

Validate your label file:

```bash
.venv/bin/python scripts/validate_labels.py --labels data/labels.csv --samples-root data/samples
```

Useful option when preparing labels before clips are copied:

```bash
.venv/bin/python scripts/validate_labels.py --labels data/labels.csv --allow-missing-files
```
