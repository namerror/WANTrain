from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from wantrain.config import ReferenceConfig, resolve_defaults, validate_reference_config
from wantrain.reference import run_reference


def test_reference_smoke_run_writes_artifacts(tmp_path: Path) -> None:
    raw = resolve_defaults(
        {
            "run": {
                "name": "pytest-smoke",
                "seed": 7,
                "device": "cpu",
                "max_epochs": 1,
                "max_steps": 2,
            },
            "workload": {
                "name": "synthetic_classification",
                "batch_size": 8,
                "num_examples": 64,
                "input_dim": 8,
                "num_classes": 3,
                "validation_split": 0.25,
                "num_workers": 0,
            },
            "optimizer": {
                "name": "sgd",
                "learning_rate": 0.01,
                "momentum": 0.0,
                "weight_decay": 0.0,
            },
            "validation": {"every_steps": 1, "target_accuracy": 0.5, "max_batches": 1},
            "telemetry": {"format": "jsonl", "output_dir": str(tmp_path / "runs")},
        }
    )
    config = ReferenceConfig(raw=raw)
    validate_reference_config(config)

    run_dir = run_reference(config)

    assert (run_dir / "resolved_config.yaml").exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "events.jsonl").exists()

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["final_step"] == 2
    assert metrics["device"] == "cpu"

    events = [
        json.loads(line)
        for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert events[0]["event_type"] == "run_started"
    assert events[-1]["event_type"] == "run_finished"
    assert {event["event_type"] for event in events} >= {
        "train_step_completed",
        "validation_completed",
    }

