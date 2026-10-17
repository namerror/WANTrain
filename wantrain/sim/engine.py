"""Event-driven shared-link simulation for the pipeline workload."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
from itertools import count

from wantrain.sim.workload import Scenario, Task, build_tasks


EPS = 1e-9
POLICIES = ("fifo", "fair", "srpt", "echelon_static", "echelon_oracle")


@dataclass(frozen=True)
class Interval:
    start_ms: float
    end_ms: float
    tasks: tuple[str, ...]


@dataclass(frozen=True)
class Result:
    policy: str
    job_completion_ms: dict[str, float]
    receiver_wait_ms: dict[str, float]
    mean_flow_completion_ms: float
    mean_flow_tardiness_ms: float
    link_busy_fraction: float
    total_bytes_mib: float
    compute_intervals: tuple[Interval, ...]
    network_intervals: tuple[Interval, ...]
    task_start_ms: dict[str, float]
    task_end_ms: dict[str, float]
    flow_release_ms: dict[str, float]
    flow_target_ms: dict[str, float]
    decisions: tuple["Decision", ...]


@dataclass(frozen=True)
class Decision:
    time_ms: float
    selected: tuple[str, ...]
    contenders: tuple[str, ...]
    key: str
    values: tuple[tuple[str, float], ...]


def simulate(scenario: Scenario, policy: str) -> Result:
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}; choose from {', '.join(POLICIES)}")

    tasks = build_tasks(scenario)
    jobs = {job.name: job for job in scenario.jobs}
    dependents: dict[str, list[str]] = {task_id: [] for task_id in tasks}
    remaining_deps = {task_id: len(task.deps) for task_id, task in tasks.items()}
    for task in tasks.values():
        for predecessor in task.deps:
            dependents[predecessor].append(task.id)

    events: list[tuple[float, int, str, str]] = []
    event_order = count()

    def post(when: float, kind: str, task_id: str) -> None:
        heapq.heappush(events, (when, next(event_order), kind, task_id))

    for job in scenario.jobs:
        post(job.start_ms, "root", f"{job.name}:F0:0")

    time_ms = min(job.start_ms for job in scenario.jobs)
    ready_compute: set[str] = set()
    worker_busy: dict[str, str] = {}
    flow_remaining: dict[str, float] = {}
    flow_release: dict[str, float] = {}
    flow_target: dict[str, float] = {}
    flow_origin: dict[tuple[str, str], float] = {}
    task_start: dict[str, float] = {}
    task_end: dict[str, float] = {}
    compute_intervals: list[Interval] = []
    network_intervals: list[Interval] = []
    decisions: list[Decision] = []
    last_selection: tuple[str, ...] = ()
    last_contenders: tuple[str, ...] = ()
    link_busy_ms = 0.0
    steps = 0

    def make_ready(task_id: str) -> None:
        task = tasks[task_id]
        if task.kind == "compute":
            ready_compute.add(task_id)
            return
        flow_remaining[task_id] = task.size_mib
        flow_release[task_id] = time_ms
        group = (task.job, task.phase)
        if task.sequence == 0:
            flow_origin[group] = time_ms
        if group not in flow_origin:
            raise RuntimeError(f"flow {task_id} arrived before its head flow")
        job = jobs[task.job]
        if task.phase == "forward":
            spacing = job.forward_ms[1] if policy == "echelon_oracle" else job.profile_forward_ms[1]
        else:
            spacing = job.backward_ms[0] if policy == "echelon_oracle" else job.profile_backward_ms[0]
        flow_target[task_id] = flow_origin[group] + task.sequence * spacing

    def finish(task_id: str) -> None:
        if task_id in task_end:
            raise RuntimeError(f"task {task_id} finished twice")
        task_end[task_id] = time_ms
        for dependent in dependents[task_id]:
            remaining_deps[dependent] -= 1
            if remaining_deps[dependent] == 0:
                make_ready(dependent)

    def start_compute() -> None:
        # A task's local predecessor edge fixes each worker's order. Sorting
        # here only makes ties deterministic if a future workload adds choices.
        for task_id in sorted(ready_compute):
            task = tasks[task_id]
            assert task.worker is not None
            if task.worker in worker_busy:
                continue
            ready_compute.remove(task_id)
            worker_busy[task.worker] = task_id
            task_start[task_id] = time_ms
            post(time_ms + task.duration_ms, "compute_done", task_id)

    def rates() -> dict[str, float]:
        if not flow_remaining:
            return {}
        if policy == "fair":
            rate = scenario.capacity_mib_per_ms / len(flow_remaining)
            return {task_id: rate for task_id in flow_remaining}
        if policy == "fifo":
            selected = min(flow_remaining, key=lambda task_id: (flow_release[task_id], task_id))
        elif policy == "srpt":
            selected = min(flow_remaining, key=lambda task_id: (flow_remaining[task_id], flow_release[task_id], task_id))
        else:
            selected = min(flow_remaining, key=lambda task_id: (flow_target[task_id], flow_remaining[task_id], task_id))
        return {selected: scenario.capacity_mib_per_ms}

    while len(task_end) < len(tasks):
        steps += 1
        if steps > 100_000:
            raise RuntimeError("simulation did not converge")

        while events and events[0][0] <= time_ms + EPS:
            _, _, kind, task_id = heapq.heappop(events)
            if kind == "root":
                make_ready(task_id)
            elif kind == "compute_done":
                task = tasks[task_id]
                assert task.worker is not None
                if worker_busy.pop(task.worker, None) != task_id:
                    raise RuntimeError(f"worker state corrupted for {task_id}")
                compute_intervals.append(Interval(task_start[task_id], time_ms, (task_id,)))
                finish(task_id)
            elif kind == "flow_delivered":
                finish(task_id)
            else:
                raise RuntimeError(f"unknown event {kind}")

        start_compute()
        if len(task_end) == len(tasks):
            break

        active_rates = rates()
        selection = tuple(sorted(active_rates))
        contenders = tuple(sorted(flow_remaining))
        if flow_remaining and (selection != last_selection or contenders != last_contenders):
            if policy == "fifo":
                key = "release_ms"
                values = tuple((task_id, flow_release[task_id]) for task_id in sorted(flow_remaining))
            elif policy == "srpt":
                key = "remaining_mib"
                values = tuple((task_id, flow_remaining[task_id]) for task_id in sorted(flow_remaining))
            elif policy == "fair":
                key = "rate_mib_per_ms"
                values = tuple((task_id, active_rates[task_id]) for task_id in sorted(flow_remaining))
            else:
                key = "target_ms"
                values = tuple((task_id, flow_target[task_id]) for task_id in sorted(flow_remaining))
            decisions.append(Decision(time_ms, selection, contenders, key, values))
        last_selection = selection
        last_contenders = contenders
        next_event = events[0][0] if events else float("inf")
        next_transfer = min(
            (time_ms + flow_remaining[task_id] / rate for task_id, rate in active_rates.items()),
            default=float("inf"),
        )
        next_time = min(next_event, next_transfer)
        if next_time == float("inf"):
            pending = [task_id for task_id in tasks if task_id not in task_end]
            raise RuntimeError(f"simulation deadlocked with pending tasks: {pending}")
        if next_time < time_ms - EPS:
            raise RuntimeError("simulation time moved backward")
        delta = max(0.0, next_time - time_ms)
        if active_rates and delta > EPS:
            network_intervals.append(Interval(time_ms, next_time, tuple(sorted(active_rates))))
            link_busy_ms += delta
            for task_id, rate in active_rates.items():
                flow_remaining[task_id] = max(0.0, flow_remaining[task_id] - rate * delta)
        time_ms = next_time

        for task_id in list(flow_remaining):
            if flow_remaining[task_id] <= EPS:
                del flow_remaining[task_id]
                post(time_ms + scenario.delivery_delay_ms, "flow_delivered", task_id)

    job_completion = {
        job.name: task_end[f"{job.name}:B0:0"] - job.start_ms for job in scenario.jobs
    }
    receiver_wait = {job.name: 0.0 for job in scenario.jobs}
    for task in tasks.values():
        if task.kind != "compute" or not (task.id.endswith(f"F1:{task.microbatch}") or task.id.endswith(f"B0:{task.microbatch}")):
            continue
        flow_predecessors = [predecessor for predecessor in task.deps if tasks[predecessor].kind == "flow"]
        if not flow_predecessors:
            continue
        local_predecessors = [predecessor for predecessor in task.deps if tasks[predecessor].kind == "compute"]
        local_ready = max([jobs[task.job].start_ms] + [task_end[predecessor] for predecessor in local_predecessors])
        receiver_wait[task.job] += max(0.0, task_end[flow_predecessors[0]] - local_ready)

    flows = [task for task in tasks.values() if task.kind == "flow"]
    mean_fct = sum(task_end[task.id] - flow_release[task.id] for task in flows) / len(flows)
    mean_tardiness = sum(task_end[task.id] - flow_target[task.id] for task in flows) / len(flows)
    horizon = max(task_end.values()) - min(job.start_ms for job in scenario.jobs)
    return Result(
        policy=policy,
        job_completion_ms=job_completion,
        receiver_wait_ms=receiver_wait,
        mean_flow_completion_ms=mean_fct,
        mean_flow_tardiness_ms=mean_tardiness,
        link_busy_fraction=link_busy_ms / horizon,
        total_bytes_mib=sum(task.size_mib for task in flows),
        compute_intervals=tuple(compute_intervals),
        network_intervals=tuple(network_intervals),
        task_start_ms=task_start,
        task_end_ms=task_end,
        flow_release_ms=flow_release,
        flow_target_ms=flow_target,
        decisions=tuple(decisions),
    )
