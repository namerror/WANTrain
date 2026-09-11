from pathlib import Path

import pytest
import yaml

from wantrain.config import ConfigError, load_reference_config, resolve_defaults


def test_resolve_defaults_adds_reference_run_defaults() -> None:
    resolved = resolve_defaults(
        {
            "run": {"name": "example"},
            "workload": {"name": "synthetic_classification", "batch_size": 4},
        }
    )

    assert resolved["schema_version"] == 1
    assert resolved["run"]["device"] == "auto"
    assert resolved["telemetry"]["output_dir"] == "runs"
    assert resolved["optimizer"]["name"] == "sgd"


def test_load_reference_config_rejects_invalid_device(tmp_path: Path) -> None:
    config_path = tmp_path / "bad.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "run": {"name": "bad", "device": "tpu"},
                "workload": {"name": "synthetic_classification", "batch_size": 4},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="run.device"):
        load_reference_config(config_path)

