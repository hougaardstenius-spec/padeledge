import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))
TRAIN_SCRIPT = BASE_DIR / "scripts" / "train_shot_model.py"
SAMPLE_BANDEJA = BASE_DIR / "data" / "samples" / "overhead" / "bandeja" / "Bandeja 2.mp4"
SAMPLE_VIBORA = BASE_DIR / "data" / "samples" / "overhead" / "vibora" / "Vibora 2.mp4"


def _prepare_tiny_dataset(root: Path) -> Path:
    data_dir = root / "data" / "samples" / "overhead"
    b_dir = data_dir / "bandeja"
    v_dir = data_dir / "vibora"
    b_dir.mkdir(parents=True, exist_ok=True)
    v_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SAMPLE_BANDEJA, b_dir / SAMPLE_BANDEJA.name)
    shutil.copy2(SAMPLE_VIBORA, v_dir / SAMPLE_VIBORA.name)
    return root / "data" / "samples"


def _train_for_test(tmp_path: Path):
    data_dir = _prepare_tiny_dataset(tmp_path)
    model_dir = tmp_path / "models"
    model_path = model_dir / "shot_classifier.pkl"
    metrics_path = model_dir / "metrics.json"
    archive_dir = model_dir / "archive"

    env = os.environ.copy()
    env["PADELEDGE_DATA_DIR"] = str(data_dir)
    env["PADELEDGE_MODEL_PATH"] = str(model_path)
    env["PADELEDGE_METRICS_PATH"] = str(metrics_path)
    env["PADELEDGE_ARCHIVE_DIR"] = str(archive_dir)

    proc = subprocess.run(
        [sys.executable, str(TRAIN_SCRIPT), "-v"],
        capture_output=True,
        text=True,
        cwd=str(BASE_DIR),
        env=env,
    )
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr
    return model_path, metrics_path


def test_training_script_creates_model_and_metrics(tmp_path):
    model_path, metrics_path = _train_for_test(tmp_path)

    assert model_path.exists()
    assert model_path.stat().st_size > 0
    assert metrics_path.exists()

    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert "feature_frames" in payload
    assert "feature_dim" in payload


def test_shot_detector_analyze_returns_valid_structure(tmp_path, monkeypatch):
    model_path, metrics_path = _train_for_test(tmp_path)
    monkeypatch.setenv("PADELEDGE_MODEL_PATH", str(model_path))
    monkeypatch.setenv("PADELEDGE_METRICS_PATH", str(metrics_path))

    import importlib
    import utils.shot_detector as shot_detector

    importlib.reload(shot_detector)
    detector = shot_detector.ShotDetector()
    preds, timestamps, keypoints = detector.analyze(str(SAMPLE_BANDEJA))

    assert isinstance(preds, list)
    assert isinstance(timestamps, list)
    assert isinstance(keypoints, list)
    assert len(preds) == len(timestamps) == len(keypoints)
    assert len(preds) > 0
    assert all(isinstance(t, float) for t in timestamps)
