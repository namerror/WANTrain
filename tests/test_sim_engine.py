"""Checks for simulator behavior that would otherwise distort conclusions."""

from pathlib import Path

from wantrain.sim.engine import POLICIES, simulate
from wantrain.sim.workload import Job, Scenario, build_tasks, load_scenario


def job(name: str, *, microbatches: int = 1, activation_mib: float = 2.0) -> Job:
    return Job(
        name=name,
        start_ms=0.0,
        microbatches=microbatches,
        forward_ms=(1.0, 1.0),
        backward_ms=(1.0, 1.0),
        profile_forward_ms=(1.0, 1.0),
        profile_backward_ms=(1.0, 1.0),
        activation_mib=activation_mib,
        gradient_mib=1.0,
    )


def test_one_microbatch_has_hand_computable_training_step() -> None:
    training_job = Job("A", 0.0, 1, (2.0, 3.0), (7.0, 5.0), (2.0, 3.0), (7.0, 5.0), 4.0, 2.0)
    scenario = Scenario(1.0, 0.0, (training_job,))
    tasks = build_tasks(scenario)
    result = simulate(scenario, "fifo")

    # F0(2) -> activation(4) -> F1(3) -> B1(5) -> gradient(2) -> B0(7).
    assert result.job_completion_ms["A"] == 23.0
    assert result.total_bytes_mib == 6.0
    for task in tasks.values():
        for predecessor in task.deps:
            if task.kind == "compute":
                assert result.task_start_ms[task.id] >= result.task_end_ms[predecessor]
            else:
                assert result.flow_release_ms[task.id] >= result.task_end_ms[predecessor]


def test_equal_sharing_delivers_simultaneous_flows_together() -> None:
    scenario = Scenario(1.0, 0.0, (job("A"), job("B")))
    fifo = simulate(scenario, "fifo")
    fair = simulate(scenario, "fair")

    assert (fifo.task_end_ms["A:A:0"], fifo.task_end_ms["B:A:0"]) == (3.0, 5.0)
    assert (fair.task_end_ms["A:A:0"], fair.task_end_ms["B:A:0"]) == (5.0, 5.0)


def test_contention_trace_exposes_different_policy_choices() -> None:
    config = Path(__file__).resolve().parents[1] / "experiments" / "configs" / "scheduling_demo.yaml"
    scenario = load_scenario(config)
    fifo = simulate(scenario, "fifo")
    srpt = simulate(scenario, "srpt")
    timing = simulate(scenario, "echelon_static")

    def choice_at_three(result):
        return next(decision for decision in result.decisions if decision.time_ms == 3.0)

    # B's 2 MiB activation arrives while A's 6 MiB activation has 5 MiB left.
    # FIFO and timing priority continue A; size-aware SRPT preempts for B.
    assert choice_at_three(fifo).selected == ("A:A:0",)
    assert choice_at_three(srpt).selected == ("B:A:0",)
    assert choice_at_three(timing).selected == ("A:A:0",)


def test_all_policies_preserve_bytes_and_are_deterministic() -> None:
    scenario = Scenario(1.0, 0.0, (job("A", microbatches=3, activation_mib=3.0), job("B", microbatches=3)))
    expected_bytes = 3 * (3.0 + 1.0 + 2.0 + 1.0)
    for policy in POLICIES:
        result = simulate(scenario, policy)
        network_time = sum(interval.end_ms - interval.start_ms for interval in result.network_intervals)
        assert result.total_bytes_mib == expected_bytes
        assert abs(network_time - expected_bytes) < 1e-7
        assert result == simulate(scenario, policy)


def test_oracle_only_changes_scheduler_visible_compute_spacing() -> None:
    actual = job("A", microbatches=2)
    stale = Job(
        actual.name, actual.start_ms, actual.microbatches,
        actual.forward_ms, actual.backward_ms,
        (1.0, 4.0), (4.0, 1.0),
        actual.activation_mib, actual.gradient_mib,
    )
    scenario = Scenario(1.0, 0.0, (stale,))
    static = simulate(scenario, "echelon_static")
    oracle = simulate(scenario, "echelon_oracle")

    assert static.flow_target_ms["A:A:1"] == static.flow_target_ms["A:A:0"] + 4.0
    assert oracle.flow_target_ms["A:A:1"] == oracle.flow_target_ms["A:A:0"] + 1.0
    assert static.total_bytes_mib == oracle.total_bytes_mib


def test_profile_error_example_changes_information_not_actual_work() -> None:
    root = Path(__file__).resolve().parents[1] / "experiments" / "configs"
    accurate = load_scenario(root / "scheduling_demo.yaml")
    perturbed = load_scenario(root / "scheduling_profile_error_demo.yaml")

    for original, changed in zip(accurate.jobs, perturbed.jobs):
        assert original.forward_ms == changed.forward_ms
        assert original.backward_ms == changed.backward_ms
        assert original.activation_mib == changed.activation_mib
        assert original.gradient_mib == changed.gradient_mib

    for policy in ("fifo", "fair", "srpt", "echelon_oracle"):
        assert simulate(accurate, policy).job_completion_ms == simulate(perturbed, policy).job_completion_ms

    original_static = simulate(accurate, "echelon_static")
    changed_static = simulate(perturbed, "echelon_static")
    assert original_static.job_completion_ms != changed_static.job_completion_ms
    assert changed_static.job_completion_ms["B"] > original_static.job_completion_ms["B"]
