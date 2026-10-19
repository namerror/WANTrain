# Scheduling policies in the current simulator

The flow simulator accepts five policy names: `fifo`, `fair`, `srpt`,
`echelon_static`, and `echelon_oracle`. They all run the same workload and
share the same link model. This page describes the rules implemented in
[`engine.py`](../wantrain/sim/engine.py), including the information each rule
uses. For the training-step dependencies and event loop, see the
[simulator walkthrough](flow_scheduling_design.md).

## Rules shared by all policies

A flow becomes **ready** when its sending compute task finishes. Ready flows
compete for one fluid link with capacity `C` MiB/ms. A completed transmission
is delivered after the configured fixed delay; it cannot be selected again
while awaiting delivery. The engine selects rates after processing events at
the current time, including job starts, compute completions, and deliveries.
It advances to the next event or transmission completion and selects rates
again. Changing service at an event has no modeled cost. Compute tasks do not
preempt, and a policy cannot change their order or duration.

The priority policies each give the **entire** link to one ready flow. Their
choice can change at an event, so a partially transmitted flow can resume
later. `fair` serves every ready flow at once. When no flow is ready, the link
is idle. All five policies use the same actual compute durations, flow sizes,
and link capacity. Only the two `echelon_*` policies use compute duration
estimates as priority inputs.

| Policy | Link allocation when flows are ready | Selection data | Decision trace key |
|---|---|---|---|
| [`fifo`](#fifo-first-in-first-out) | Full capacity to one flow | Flow release time, then flow ID | `release_ms` |
| [`fair`](#fair-equal-per-flow-sharing) | `C / N` to each of `N` ready flows | Number of ready flows | `rate_mib_per_ms` |
| [`srpt`](#srpt-shortest-remaining-processing-time) | Full capacity to one flow | Remaining MiB, then release time and flow ID | `remaining_mib` |
| [`echelon_static`](#echelon_static-profile-based-timing-priority) | Full capacity to one flow | Profile-based target, then remaining MiB and flow ID | `target_ms` |
| [`echelon_oracle`](#echelon_oracle-true-profile-information-control) | Full capacity to one flow | Actual-duration target, then remaining MiB and flow ID | `target_ms` |

Flow ID is the stable final tie-breaker; for example, `A:A:1` is Job A's
activation for microbatch 1, and `A:G:1` is its gradient. `fair` has no
tie-break because it serves all ready flows equally.

## `fifo`: first in, first out

Select the ready flow with the earliest **release time**. If several flows
were released at exactly the same time, select the lexicographically smallest
flow ID. Give that flow the whole link until the next event, then apply the
same rule again. A newly released flow with a later release time cannot
displace an older ready flow, even if it is much smaller or its receiver is
waiting. FIFO uses neither flow size nor compute profiles for priority.

This is the arrival-order baseline. In the demo, `A:A:0` is released at
2 ms. When `B:A:0` arrives at 3 ms, FIFO continues `A:A:0` because its
release time is earlier. The `release_ms` values in `--decisions` show why.

## `fair`: equal per-flow sharing

If `N` flows are ready, assign **each flow** `C / N` MiB/ms. Recalculate the
division when a flow arrives or finishes. With one ready flow it gets the
whole link. Sharing is per flow, not per job: a job with more ready flows can
receive more aggregate capacity. Fair sharing does not inspect release order,
remaining size, or compute profiles to assign rates.

This baseline shows what happens when all ready transfers progress together
instead of choosing one. At 3 ms in the demo, the 1 MiB/ms link has two
ready flows, so each receives 0.5 MiB/ms. The decision trace lists both
flows under `selected` and gives their individual `rate_mib_per_ms` values.

## `srpt`: shortest remaining processing time

Select the ready flow with the fewest **remaining MiB** and give it the whole
link. Break equal-size ties by earlier release time, then flow ID. The engine
deducts bytes as a flow is served, so the comparison uses its current
remaining size, not its original size. Since all flows would use the same
link capacity when selected, ordering by remaining MiB is equivalent to
ordering by remaining transmission time. A new, shorter flow can preempt the
current one at an event. SRPT uses no compute profiles or receiver timing.

This is the size-aware network baseline. At 3 ms in the demo, `A:A:0` has
5 MiB left and the new `B:A:0` has 2 MiB; SRPT switches to B. The trace key
`remaining_mib` shows the values at each decision. Favouring short transfers
can improve flow completion while delaying a large transfer that gates a
training stage, so compare both jobs' step times as well as mean flow time.

## Timing targets used by the two `echelon_*` policies

The simulator groups each job's activation flows as one forward sequence and
its gradient flows as one backward sequence. Within each job and phase, the
**first flow's release time** is the origin `r`. The first flow has sequence
number `0`; later flows have `1`, `2`, and so on. Each flow receives a fixed
target when it becomes ready:

```text
target_ms = r + sequence_number × receiving_stage_time_ms
```

For activations, `receiving_stage_time_ms` is the job's stage-1 forward
duration (`forward_ms[1]` or `profile_forward_ms[1]`). For gradients, it is
the stage-0 backward duration (`backward_ms[0]` or
`profile_backward_ms[0]`). Gradient sequence numbers follow **reverse
microbatch order**, matching the backward phase. Each phase gets its own
origin. The target is a priority key expressing an estimated spacing between
receiver needs; it is not a guaranteed deadline or a calculation of the
receiver's exact ready time. Once assigned, a target is not revised.

Both timing policies pick the ready flow with the **smallest target**. Equal
targets are broken by fewer remaining MiB, then flow ID. They give the whole
link to that flow and reconsider at the next event. An arriving flow with an
earlier target can preempt the current flow. These are simple earliest-target
heuristics inspired by [EchelonFlow's staggered finish-time
idea](https://conferences.sigcomm.org/hotnets/2022/papers/hotnets22_pan.pdf);
they do not implement that paper's optimization or system.

## `echelon_static`: profile-based timing priority

Compute the targets using the job's **scheduler-visible**
`profile_forward_ms[1]` and `profile_backward_ms[0]`. If those optional YAML
fields are absent, the scenario loader copies the corresponding actual
durations into the profile. Its target keys use the profile estimates;
actual durations determine when compute tasks and flows become ready.

This is the timing-aware policy under study. Its purpose is to test whether
receiver compute spacing helps schedule transfers and how a wrong profile
changes that choice. In the basic demo, the profiles default to actual
durations. In
[`scheduling_profile_error_demo.yaml`](../experiments/configs/scheduling_profile_error_demo.yaml),
Job B's visible stage-1 forward and stage-0 backward times are doubled while
its actual work stays fixed; only this policy's choices can change because
of those profile fields.

## `echelon_oracle`: true-profile information control

Use **actual** `forward_ms[1]` and `backward_ms[0]` to compute the same kind
of targets, then apply exactly the same earliest-target rule and tie-breaks
as `echelon_static`. The policy is given these durations counterfactually to
isolate the effect of profile error. It does not know future flow arrivals,
and it is **not** an optimal schedule or a performance upper bound.

When a config omits `profile_*` fields, static and oracle targets coincide,
as they do in the basic demo. If only visible profiles change, oracle's
choices and actual workload remain the same. Compare static with oracle on
paired configs to study the information difference, and compare both with
FIFO, fair sharing, and SRPT to understand the scheduling tradeoff.

## Inspect a decision

Run the same scenario under all policies and print their choices:

```bash
uv run python -m wantrain.sim.run \
  --config experiments/configs/scheduling_demo.yaml \
  --decisions --no-timeline
```

At `t = 3 ms`, the demo has `A:A:0` (5 MiB remaining, released at 2 ms,
target 2 ms) and `B:A:0` (2 MiB, released at 3 ms, target 3 ms). FIFO
chooses A by release time; SRPT chooses B by remaining size; both timing
policies choose A by target. Fair sharing gives each half the link. A trace
records the current contenders, the selected flow or flows, and the displayed
comparison key. The trace prints the primary key; it does not print every
tie-break value. The JSON includes `flow_target_ms` even for FIFO, fair
sharing, and SRPT as a diagnostic, but those policies do not use targets to
select flows. For exact link-service intervals and task completion times, use
`--output temp/demo.json`; see the [experiment guide](experiments.md) for
metric definitions. The demo is a crafted example of mechanism, not evidence
that one policy generally wins.
