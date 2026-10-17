"""A two-stage, fill-and-drain pipeline training workload.

The nodes describe dependencies, not actual PyTorch or NCCL execution. Each job
owns two workers; different jobs share the simulated network link.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Job:
    name: str
    start_ms: float
    microbatches: int
    forward_ms: tuple[float, float]
    backward_ms: tuple[float, float]
    profile_forward_ms: tuple[float, float]
    profile_backward_ms: tuple[float, float]
    activation_mib: float
    gradient_mib: float


@dataclass(frozen=True)
class Scenario:
    capacity_mib_per_ms: float
    delivery_delay_ms: float
    jobs: tuple[Job, ...]


@dataclass(frozen=True)
class Task:
    id: str
    job: str
    kind: str  # "compute" or "flow"
    phase: str  # "forward" or "backward"
    microbatch: int
    deps: tuple[str, ...]
    worker: str | None = None
    duration_ms: float = 0.0
    size_mib: float = 0.0
    sequence: int = 0  # transfer order within a job and phase


def _positive(value: object, label: str, *, allow_zero: bool = False) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not (number >= 0 if allow_zero else number > 0) or number == float("inf"):
        raise ValueError(f"{label} must be finite and {'non-negative' if allow_zero else 'positive'}")
    return number


def _pair(value: object, label: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{label} must contain two stage durations")
    return (_positive(value[0], f"{label}[0]"), _positive(value[1], f"{label}[1]"))


def load_scenario(path: Path) -> Scenario:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("scenario must be a YAML mapping")
    link = raw.get("link")
    if not isinstance(link, dict):
        raise ValueError("link must be a mapping")
    capacity = _positive(link.get("capacity_mib_per_ms"), "link.capacity_mib_per_ms")
    delay = _positive(link.get("delivery_delay_ms", 0), "link.delivery_delay_ms", allow_zero=True)
    raw_jobs = raw.get("jobs")
    if not isinstance(raw_jobs, list) or not raw_jobs:
        raise ValueError("jobs must be a nonempty list")
    jobs: list[Job] = []
    for index, raw_job in enumerate(raw_jobs):
        if not isinstance(raw_job, dict):
            raise ValueError(f"jobs[{index}] must be a mapping")
        name = raw_job.get("name")
        if not isinstance(name, str) or not name or ":" in name:
            raise ValueError(f"jobs[{index}].name must be a nonempty string without ':'")
        count = raw_job.get("microbatches")
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError(f"jobs[{index}].microbatches must be a positive integer")
        forward = _pair(raw_job.get("forward_ms"), f"jobs[{index}].forward_ms")
        backward = _pair(raw_job.get("backward_ms"), f"jobs[{index}].backward_ms")
        jobs.append(
            Job(
                name=name,
                start_ms=_positive(raw_job.get("start_ms", 0), f"jobs[{index}].start_ms", allow_zero=True),
                microbatches=count,
                forward_ms=forward,
                backward_ms=backward,
                profile_forward_ms=_pair(raw_job.get("profile_forward_ms", list(forward)), f"jobs[{index}].profile_forward_ms"),
                profile_backward_ms=_pair(raw_job.get("profile_backward_ms", list(backward)), f"jobs[{index}].profile_backward_ms"),
                activation_mib=_positive(raw_job.get("activation_mib"), f"jobs[{index}].activation_mib"),
                gradient_mib=_positive(raw_job.get("gradient_mib"), f"jobs[{index}].gradient_mib"),
            )
        )
    if len({job.name for job in jobs}) != len(jobs):
        raise ValueError("job names must be unique")
    return Scenario(capacity, delay, tuple(jobs))


def build_tasks(scenario: Scenario) -> dict[str, Task]:
    tasks: dict[str, Task] = {}

    def add(task: Task) -> None:
        if task.id in tasks:
            raise ValueError(f"duplicate task {task.id}")
        tasks[task.id] = task

    for job in scenario.jobs:
        name = job.name
        count = job.microbatches
        for m in range(count):
            f0 = f"{name}:F0:{m}"
            activation = f"{name}:A:{m}"
            f1 = f"{name}:F1:{m}"
            add(Task(f0, name, "compute", "forward", m,
                     (f"{name}:F0:{m - 1}",) if m else (),
                     f"{name}:S0", job.forward_ms[0]))
            add(Task(activation, name, "flow", "forward", m, (f0,),
                     size_mib=job.activation_mib, sequence=m))
            add(Task(f1, name, "compute", "forward", m,
                     (activation, f"{name}:F1:{m - 1}") if m else (activation,),
                     f"{name}:S1", job.forward_ms[1]))

        # GPipe-style fill-and-drain: all stage-1 forwards finish before the
        # first backward. Each stage then processes microbatches in reverse.
        for m in reversed(range(count)):
            b1 = f"{name}:B1:{m}"
            gradient = f"{name}:G:{m}"
            b0 = f"{name}:B0:{m}"
            b1_deps = (f"{name}:F1:{count - 1}",) if m == count - 1 else (f"{name}:B1:{m + 1}",)
            b0_deps = (gradient, f"{name}:F0:{count - 1}") if m == count - 1 else (gradient, f"{name}:B0:{m + 1}")
            add(Task(b1, name, "compute", "backward", m, b1_deps,
                     f"{name}:S1", job.backward_ms[1]))
            add(Task(gradient, name, "flow", "backward", m, (b1,),
                     size_mib=job.gradient_mib, sequence=count - 1 - m))
            add(Task(b0, name, "compute", "backward", m, b0_deps,
                     f"{name}:S0", job.backward_ms[0]))

    return tasks
