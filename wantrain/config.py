from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a run configuration is missing required or valid values."""


@dataclass(frozen=True)
class ReferenceConfig:
    raw: dict[str, Any]

    @property
    def run_name(self) -> str:
        return str(self.raw["run"]["name"])

    @property
    def seed(self) -> int:
        return int(self.raw["run"]["seed"])

    @property
    def output_dir(self) -> Path:
        return Path(self.raw["telemetry"]["output_dir"])

    @property
    def device(self) -> str:
        return str(self.raw["run"].get("device", "auto"))


def load_reference_config(path: Path) -> ReferenceConfig:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a YAML mapping")
    config = ReferenceConfig(raw=resolve_defaults(data))
    validate_reference_config(config)
    return config


def resolve_defaults(data: dict[str, Any]) -> dict[str, Any]:
    resolved = deepcopy(data)
    resolved.setdefault("schema_version", 1)
    resolved.setdefault("run", {})
    resolved["run"].setdefault("seed", 1234)
    resolved["run"].setdefault("device", "auto")
    resolved["run"].setdefault("max_epochs", 1)
    resolved["run"].setdefault("max_steps", None)
    resolved["run"].setdefault("max_duration", None)

    resolved.setdefault("workload", {})
    resolved["workload"].setdefault("data_dir", "data")
    resolved["workload"].setdefault("num_workers", 2)
    resolved["workload"].setdefault("validation_split", 0.1)
    resolved["workload"].setdefault("download", True)

    resolved.setdefault("optimizer", {})
    resolved["optimizer"].setdefault("name", "sgd")
    resolved["optimizer"].setdefault("learning_rate", 0.1)
    resolved["optimizer"].setdefault("momentum", 0.9)
    resolved["optimizer"].setdefault("weight_decay", 0.0)

    resolved.setdefault("validation", {})
    resolved["validation"].setdefault("every_steps", 100)
    resolved["validation"].setdefault("target_accuracy", None)
    resolved["validation"].setdefault("max_batches", None)

    resolved.setdefault("telemetry", {})
    resolved["telemetry"].setdefault("format", "jsonl")
    resolved["telemetry"].setdefault("output_dir", "runs")
    return resolved


def validate_reference_config(config: ReferenceConfig) -> None:
    data = config.raw
    required_sections = ["run", "workload", "optimizer", "validation", "telemetry"]
    for section in required_sections:
        if not isinstance(data.get(section), dict):
            raise ConfigError(f"missing or invalid '{section}' section")

    if data["schema_version"] != 1:
        raise ConfigError("schema_version must be 1")
    if not data["run"].get("name"):
        raise ConfigError("run.name is required")
    if config.device not in {"auto", "cpu", "cuda"}:
        raise ConfigError("run.device must be one of: auto, cpu, cuda")
    if int(data["run"]["max_epochs"]) < 1:
        raise ConfigError("run.max_epochs must be at least 1")
    if data["run"]["max_steps"] is not None and int(data["run"]["max_steps"]) < 1:
        raise ConfigError("run.max_steps must be at least 1 when set")

    workload_name = data["workload"].get("name")
    if workload_name not in {"synthetic_classification", "cifar10_resnet18"}:
        raise ConfigError("workload.name must be synthetic_classification or cifar10_resnet18")
    if int(data["workload"].get("batch_size", 0)) < 1:
        raise ConfigError("workload.batch_size must be at least 1")
    validation_split = float(data["workload"]["validation_split"])
    if not 0.0 < validation_split < 1.0:
        raise ConfigError("workload.validation_split must be between 0 and 1")

    if data["optimizer"]["name"] != "sgd":
        raise ConfigError("optimizer.name must be sgd for milestone 1")
    if float(data["optimizer"]["learning_rate"]) <= 0:
        raise ConfigError("optimizer.learning_rate must be positive")
    if int(data["validation"]["every_steps"]) < 1:
        raise ConfigError("validation.every_steps must be at least 1")
    target = data["validation"]["target_accuracy"]
    if target is not None and not 0.0 < float(target) <= 1.0:
        raise ConfigError("validation.target_accuracy must be in (0, 1] when set")


def write_resolved_config(config: ReferenceConfig, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.raw, handle, sort_keys=True)

