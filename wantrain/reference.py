from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from wantrain.config import ReferenceConfig, write_resolved_config
from wantrain.telemetry import JsonlEventWriter
from wantrain.workloads import build_workload


def run_reference(config: ReferenceConfig) -> Path:
    '''
    Run a reference training job.
    '''
    seed_everything(config.seed)
    run_id = make_run_id(config.run_name)
    run_dir = config.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    write_resolved_config(config, run_dir / "resolved_config.yaml")

    device = resolve_device(config.device)
    workload = build_workload(config.raw["workload"], config.seed)
    model = workload.model.to(device)
    optimizer = build_optimizer(model, config.raw["optimizer"])
    criterion = nn.CrossEntropyLoss()
    max_duration_seconds = parse_duration_seconds(config.raw["run"].get("max_duration"))

    metrics: dict[str, Any] = {
        "run_id": run_id,
        "run_name": config.run_name,
        "device": str(device),
        "target_reached": False,
        "target_step": None,
        "best_validation_accuracy": 0.0,
        "final_step": 0,
    }

    with JsonlEventWriter(run_dir / "events.jsonl", run_id=run_id) as events:
        events.emit(
            "run_started",
            run_name=config.run_name,
            seed=config.seed,
            device=str(device),
            cuda_available=torch.cuda.is_available(),
            workload=config.raw["workload"]["name"],
        )
        start = time.monotonic()
        global_step = 0
        stop_reason = "max_epochs"
        for epoch in range(1, int(config.raw["run"]["max_epochs"]) + 1):
            progress = tqdm(workload.train_loader, desc=f"epoch {epoch}", leave=False)
            for inputs, labels in progress:
                if max_duration_seconds is not None and time.monotonic() - start >= max_duration_seconds:
                    stop_reason = "max_duration"
                    break
                if config.raw["run"]["max_steps"] is not None and global_step >= int(
                    config.raw["run"]["max_steps"]
                ):
                    stop_reason = "max_steps"
                    break

                global_step += 1
                step_start = time.monotonic()
                loss = train_step(model, optimizer, criterion, inputs, labels, device)
                step_seconds = time.monotonic() - step_start
                examples_per_second = inputs.size(0) / max(step_seconds, 1e-12)
                progress.set_postfix(loss=f"{loss:.4f}")
                events.emit(
                    "train_step_completed",
                    epoch=epoch,
                    step=global_step,
                    training_loss=loss,
                    examples_processed=inputs.size(0),
                    examples_per_second=examples_per_second,
                )

                if global_step % int(config.raw["validation"]["every_steps"]) == 0:
                    validation = validate(
                        model,
                        workload.validation_loader,
                        criterion,
                        device,
                        max_batches=config.raw["validation"]["max_batches"],
                    )
                    update_validation_metrics(metrics, validation, global_step, config.raw["validation"])
                    events.emit("validation_completed", step=global_step, **validation)
                    if metrics["target_reached"]:
                        stop_reason = "target_reached"
                        break
            if stop_reason != "max_epochs":
                break

        if global_step and stop_reason in {"max_epochs", "max_steps", "max_duration"}:
            validation = validate(
                model,
                workload.validation_loader,
                criterion,
                device,
                max_batches=config.raw["validation"]["max_batches"],
            )
            update_validation_metrics(metrics, validation, global_step, config.raw["validation"])
            events.emit("validation_completed", step=global_step, **validation)
            if metrics["target_reached"]:
                stop_reason = "target_reached"

        metrics["final_step"] = global_step
        metrics["elapsed_seconds"] = time.monotonic() - start
        metrics["stop_reason"] = stop_reason
        events.emit("run_finished", **metrics)

    with (run_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return run_dir


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(preference: str) -> torch.device:
    '''
    Either CPU or CUDA device is returned based on the preference and availability.
    '''
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("run.device is cuda but PyTorch does not report CUDA availability")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_optimizer(model: nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    return torch.optim.SGD(
        model.parameters(),
        lr=float(config["learning_rate"]),
        momentum=float(config["momentum"]),
        weight_decay=float(config["weight_decay"]),
    )


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
) -> float:
    '''
    Run a single training step: forward pass, loss computation, backward pass, and optimizer step.
    Returns the training loss as a float.
    '''

    model.train()
    inputs = inputs.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    optimizer.zero_grad(set_to_none=True)
    logits = model(inputs)
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu().item())


@torch.inference_mode()
def validate(
    model: nn.Module,
    validation_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    max_batches: int | None,
) -> dict[str, float | int]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    total_batches = 0
    for inputs, labels in validation_loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(inputs)
        loss = criterion(logits, labels)
        total_loss += float(loss.detach().cpu().item()) * inputs.size(0)
        total_correct += int((logits.argmax(dim=1) == labels).sum().detach().cpu().item())
        total_examples += inputs.size(0)
        total_batches += 1
        if max_batches is not None and total_batches >= int(max_batches):
            break
    return {
        "validation_loss": total_loss / max(total_examples, 1),
        "validation_accuracy": total_correct / max(total_examples, 1),
        "validation_examples": total_examples,
        "validation_batches": total_batches,
    }


def update_validation_metrics(
    metrics: dict[str, Any],
    validation: dict[str, float | int],
    step: int,
    validation_config: dict[str, Any],
) -> None:
    accuracy = float(validation["validation_accuracy"])
    if accuracy > float(metrics["best_validation_accuracy"]):
        metrics["best_validation_accuracy"] = accuracy
        metrics["best_step"] = step
    target = validation_config["target_accuracy"]
    if target is not None and accuracy >= float(target) and not metrics["target_reached"]:
        metrics["target_reached"] = True
        metrics["target_step"] = step


def make_run_id(run_name: str) -> str:
    '''
    Generates a unique run ID based on the current timestamp, a slugified version of the run name, and a suffix derived from the current time in nanoseconds.
    The run ID format is: YYYYMMDD-HHMMSS-slug-suffix
    '''
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    slug = "".join(ch if ch.isalnum() else "-" for ch in run_name.lower()).strip("-")
    suffix = f"{time.monotonic_ns() % 1_000_000:06d}"
    return f"{timestamp}-{slug}-{suffix}"


def parse_duration_seconds(value: str | int | float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    value = value.strip().lower()
    if value.endswith("ms"):
        return float(value[:-2]) / 1000
    if value.endswith("s"):
        return float(value[:-1])
    if value.endswith("m"):
        return float(value[:-1]) * 60
    if value.endswith("h"):
        return float(value[:-1]) * 3600
    return float(value)
