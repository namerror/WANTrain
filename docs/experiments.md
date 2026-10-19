# Running and interpreting simulator experiments

This guide covers the **flow simulator** in `wantrain/sim/`. The single-process PyTorch reference uses different config fields and does not participate in these scheduling comparisons.

For the exact selection and bandwidth rules behind each policy name, see the [scheduling policy reference](scheduling_policies.md).

## Run the two existing examples

```bash
uv run python -m wantrain.sim.run --config experiments/configs/scheduling_demo.yaml
uv run python -m wantrain.sim.run --config experiments/configs/scheduling_profile_error_demo.yaml
```

Use `--policy` more than once to compare a subset, `--decisions` to print choices whenever the ready-flow set or selected flow changes, and `--no-timeline` for a compact table. `--output temp/demo.json` writes the resolved scenario, metrics, task times, exact compute/link intervals, and decisions. The `temp/` directory is ignored by Git.

Use `--plot temp/demo.svg` to save a time-scaled SVG timeline. It places stage compute, activation link service, and gradient link service on one millisecond axis for every selected policy. Blue means forward compute, purple backward compute, green activation transfer, and orange gradient transfer. Bar labels identify phases and microbatches; hover for exact intervals. A gap on a transfer row means the shared link is serving something else or idle; simultaneous bars across transfer rows mean fair sharing. The bars show link service time, not the entire wait from flow release to delivery.

```bash
uv run python -m wantrain.sim.run --config experiments/configs/scheduling_demo.yaml \
  --policy fifo --policy srpt --decisions --output temp/flow_demo.json
```

The ASCII timeline uses the same time horizon for every policy in one command. Each column samples a small time interval, so a short task may disappear visually. Use the JSON intervals for exact timing.

## Scenario format

A simulator config has one `link` and a nonempty list of `jobs`:

```yaml
link:
  capacity_mib_per_ms: 1.0
  delivery_delay_ms: 0.0
jobs:
  - name: A
    start_ms: 0.0
    microbatches: 4
    forward_ms: [2.0, 2.0]
    backward_ms: [2.0, 2.0]
    activation_mib: 6.0
    gradient_mib: 6.0
```

| Field | Meaning |
|---|---|
| `capacity_mib_per_ms` | Total service rate of one shared fluid link. All jobs use this bottleneck. |
| `delivery_delay_ms` | Fixed delay after transmission finishes and before the receiver can consume a flow. |
| `name`, `start_ms` | Job identifier and release time. Names must be unique and cannot contain `:`. |
| `microbatches` | Positive number of pipeline microbatches in one modeled training step. |
| `forward_ms: [S0, S1]` | **Actual** forward compute duration per microbatch on stages 0 and 1. |
| `backward_ms: [S0, S1]` | **Actual** backward compute duration per microbatch on stages 0 and 1. |
| `activation_mib`, `gradient_mib` | Size of each activation or gradient flow for that job. |
| `profile_forward_ms`, `profile_backward_ms` | Optional scheduler-visible duration pairs. If omitted, each equals its actual pair. Only the timing-aware static policy uses the receiving-stage estimates for priorities. |

All time values are milliseconds; sizes are mebibytes (MiB, powers of two); capacity is MiB/ms. The example numbers are synthetic and are not measured network or GPU specifications. Within one job, every microbatch currently has the same task durations and flow sizes. No randomness is used by these examples.

## Read the output

The table prints one row per policy. In the timeline, `A:S0` means Job A's stage-0 worker. `F` and `B` mean forward and backward compute. The `link` row uses uppercase job initials for activation flows, lowercase for gradient flows, `*` when fair sharing serves multiple flows, and `.` for no sampled activity. This compact legend is clearest with the provided A/B job names; exact flow IDs are in the JSON and decision log.

| Result field | Definition | How to use it |
|---|---|---|
| `job_completion_ms` / `A step` | End of that job's final `B0:0` task minus its start time. | Primary per-job completion measure. Compare both jobs; an average can hide a harmed job. |
| `receiver_wait_ms` / `A wait*` | For each receiving `F1` or `B0` task, time from the end of its local predecessor until its required flow is delivered, clipped at zero; summed over tasks. | Diagnoses receiver dependency delays. **Includes upstream compute and pipeline ordering**, so it is not pure network stall. It is a sum of worker-milliseconds and can exceed elapsed step time. |
| `mean_flow_completion_ms` / `mean flow` | Mean time from flow release until delivery, including fixed delivery delay. | Network-level diagnostic; a lower value need not yield a shorter training step. |
| `link_busy_fraction` / `link busy` | Time the link sends at least one flow divided by the full simulated horizon from first job start to last task end. | Shows how much of the run has link work. A lower fraction can result from a longer compute or idle tail. |
| `total_bytes_mib` | Sum of generated activation and gradient flow sizes. | Accounting check; policy changes should not change this total. |
| `mean_flow_tardiness_ms` | Mean delivered time minus each flow's arrangement target. | Diagnostic in JSON. Targets depend on the visible profile, so do not directly compare this value across scenarios or between static and true-profile variants when those targets differ. |

The JSON also records `task_start_ms`, `task_end_ms`, `flow_release_ms`, `flow_target_ms`, exact `compute_intervals`, exact `network_intervals`, and `decisions`. A `Decision` lists its time, all ready contenders, selected flow or flows, and the policy key used to compare them.

## Two controlled comparisons

**Policy comparison:** use `scheduling_demo.yaml`. At 3 ms, A's activation has 5 MiB left and B's 2 MiB activation arrives. FIFO keeps A because A arrived first. SRPT gives B the link because it is smaller. The timing-aware policy keeps A because A's arrangement target is earlier. Follow the effect on the stage-1 rows before judging the final step times. In this crafted example, static timing priority finishes A/B at 60/77 ms, while SRPT finishes them at 70/59 ms. The tradeoff matters more than declaring a winner from one case.

**Information comparison:** use `scheduling_profile_error_demo.yaml`. It changes only Job B's visible receiving-stage estimates: forward stage 1 from 6 to 12 ms and backward stage 0 from 3 to 6 ms. Actual durations, transfer sizes, and link stay fixed. FIFO, fair sharing, SRPT, and the true-profile information control therefore have unchanged completion times. Here, static timing priority moves B's step from 77 to 93 ms. That demonstrates **sensitivity in a selected example**, not an average effect or a general performance claim. The true-profile variant is not a globally optimal scheduler.

## Add a scenario without losing the comparison

1. Copy one simulator YAML file under `experiments/configs/` and give it a descriptive name.
2. Change one experimental factor at a time: link capacity, actual compute ratio, flow sizes, or visible-profile error.
3. Run all policies on that same file. For profile-error pairs, preserve every actual duration and flow size and change only `profile_*` values.
4. Save exact JSON output and record the config and code version before making a figure.
5. Inspect at least one decision trace and both job completion times. Report cases where the timing-aware policy loses or ties as well as where it wins.

The next milestone is an automated **paired sweep** with a high-capacity control and an interpretable figure. There is no sweep runner or published benchmark result in this repository yet. Results from these two YAML files should be described as demonstrations of mechanism.

## Model limits

The engine has one fluid bottleneck link, fixed capacity, and free preemption at events. It has no packet queue, transport protocol, NIC, route, topology, collective, GPU kernel overlap, memory pressure, scheduler overhead, model update, or accuracy measurement. Deliberately changing those assumptions is future research work; it should not be silently inferred from the current numbers.
