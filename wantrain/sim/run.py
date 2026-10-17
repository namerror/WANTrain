"""Compare network policies on the same synthetic pipeline workload."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from wantrain.sim.engine import POLICIES, Interval, Result, simulate
from wantrain.sim.workload import Scenario, load_scenario


def _row(intervals: tuple[Interval, ...], horizon: float, width: int, label) -> str:
    cells: list[str] = []
    for index in range(width):
        moment = horizon * (index + 0.5) / width
        matching = [interval for interval in intervals if interval.start_ms <= moment < interval.end_ms]
        cells.append(label(matching) if matching else ".")
    return "".join(cells)


def timeline(result: Result, scenario: Scenario, horizon: float, width: int = 72) -> str:
    """Render a compact teaching view; JSON retains exact interval times."""
    marks = [" " for _ in range(width)]
    for tick in range(0, int(horizon) + 1, 10):
        position = min(width - 1, round(tick / horizon * (width - 1)))
        for offset, char in enumerate(str(tick)):
            if position + offset < width:
                marks[position + offset] = char
    lines = [f"{result.policy}: 0 to {horizon:.1f} ms ({horizon / width:.2f} ms/column)",
             f"  {'ms':<5} {''.join(marks)}"]
    for job in scenario.jobs:
        for stage in range(2):
            worker = f"{job.name}:S{stage}"
            intervals = tuple(
                interval for interval in result.compute_intervals
                if interval.tasks[0].startswith(f"{job.name}:")
                and interval.tasks[0].split(":")[1].endswith(str(stage))
            )
            row = _row(intervals, horizon, width,
                       lambda matches: "F" if ":F" in matches[0].tasks[0] else "B")
            lines.append(f"  {worker:<5} {row}")

    def flow_label(matches: list[Interval]) -> str:
        active = matches[0].tasks
        if len(active) > 1:
            return "*"
        flow = active[0]
        return flow[0].upper() if ":A:" in flow else flow[0].lower()

    lines.append(f"  {'link':<5} {_row(result.network_intervals, horizon, width, flow_label)}")
    lines.append("  F/B = forward/backward compute; A/B = forward flows by job; a/b = backward flows; * = fair sharing")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--policy", choices=POLICIES, action="append", help="repeat to select policies")
    parser.add_argument("--no-timeline", action="store_true")
    parser.add_argument("--decisions", action="store_true", help="show flow choices when contenders or selection change")
    parser.add_argument("--output", type=Path, help="save exact result data as JSON")
    args = parser.parse_args()

    scenario = load_scenario(args.config)
    policies = args.policy or POLICIES
    results = [simulate(scenario, policy) for policy in policies]
    job_names = [job.name for job in scenario.jobs]
    header = f"{'policy':<19}" + "".join(f"{name + ' step':>12}{name + ' wait*':>12}" for name in job_names) + f"{'mean flow':>12}{'link busy':>12}"
    print("Synthetic two-stage, fill-and-drain pipeline; shared fluid link; zero preemption cost")
    print(header)
    for result in results:
        line = f"{result.policy:<19}"
        line += "".join(
            f"{result.job_completion_ms[name]:>12.2f}{result.receiver_wait_ms[name]:>12.2f}"
            for name in job_names
        )
        line += f"{result.mean_flow_completion_ms:>12.2f}{result.link_busy_fraction:>11.1%}"
        print(line)
    print(
        "* wait is summed receiver-stage dependency wait in worker-ms; "
        "it includes upstream compute and can exceed elapsed step time."
    )
    if not args.no_timeline:
        print()
        common_horizon = max(max(result.task_end_ms.values()) for result in results)
        for result in results:
            print(timeline(result, scenario, common_horizon))
            print()
    if args.decisions:
        for result in results:
            print(f"{result.policy} scheduling decisions:")
            for decision in result.decisions:
                contenders = ", ".join(
                    f"{task_id} ({value:.2f})" for task_id, value in decision.values
                )
                print(f"  t={decision.time_ms:6.2f}: serve {', '.join(decision.selected)}; {decision.key}: {contenders}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": str(args.config),
            "scenario": asdict(scenario),
            "results": [asdict(result) for result in results],
        }
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Exact intervals and metrics: {args.output}")


if __name__ == "__main__":
    main()
