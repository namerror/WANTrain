# Simulator walkthrough: from a training step to a link decision

This is a small **synthetic simulator**, designed to make the scheduling decision visible before adding real training or network emulation. It is an adjacent experiment inspired by [EchelonFlow](https://conferences.sigcomm.org/hotnets/2022/papers/hotnets22_pan.pdf), not a reproduction of that paper's scheduler or results. Start with the [README](../README.md) for project status and the [experiment guide](experiments.md) for config fields and metrics.

## Follow the code path

```text
experiments/configs/scheduling_demo.yaml
              │
              ▼
run.py: load_scenario() ──► workload.py: Scenario and Job
              │                         │
              │                         ▼
              │                   build_tasks(): dependency graph
              │                         │
              ▼                         ▼
       engine.py: simulate(scenario, policy)
              │
              ├── advance worker compute and shared-link transfers
              ├── select ready flows using one policy
              └── record decisions, intervals, and metrics
              │
              ▼
run.py: comparison table, sampled timeline, optional exact JSON and SVG plot
```

Read [`wantrain/sim/run.py`](../wantrain/sim/run.py) first to see the input/output contract, then [`workload.py`](../wantrain/sim/workload.py) to understand task dependencies, and finally [`engine.py`](../wantrain/sim/engine.py) to trace one event. The simulator does not call `wantrain/reference.py`; the local PyTorch reference is a separate path.

## Run the first example

```bash
python3 -m wantrain.sim.run --config experiments/configs/scheduling_demo.yaml
```

Each policy runs on the same two training jobs. The table gives each job's step completion time, summed receiver-stage dependency wait, mean flow completion time, and fraction of the simulated time horizon when the link is busy. Under the table, each policy has the **same timeline scale**. `F` and `B` are forward and backward compute on a worker. Uppercase letters on the link are activation transfers; lowercase letters are gradient transfers. `*` means several flows share the link. A dot means no activity in that row. The ASCII picture samples time, so use JSON for exact intervals:

```bash
python3 -m wantrain.sim.run --config experiments/configs/scheduling_demo.yaml --output temp/scheduling_demo.json
```

To see the decisions behind one timeline:

```bash
python3 -m wantrain.sim.run --config experiments/configs/scheduling_demo.yaml --policy echelon_static --decisions
```

At `t = 3 ms`, B's 2 MiB activation arrives while A's activation has 5 MiB left. Static timing priority and FIFO keep serving A; shortest remaining flow switches to B. Inspect the recorded targets, release times, and remaining sizes with `--decisions`, then look for the downstream change on B's stage-1 worker row.

## Workload and event rules

Each job owns two workers, `S0` and `S1`, and splits one training step into microbatches. For microbatch `m`, forward work is `F0[m] → activation[m] → F1[m]`. After all stage-1 forward tasks finish, backward work runs in reverse microbatch order: `B1[m] → gradient[m] → B0[m]`. Each worker performs one compute task at a time. A task starts only when all predecessors finish. Jobs have separate workers but contend for one shared network link.

Task IDs have the form `job:kind:microbatch`: `A:F0:1` is Job A's stage-0 forward compute for microbatch 1; `A:A:1` is its activation flow; `A:G:1` is its backward gradient flow. With two microbatches, `F0:0` precedes `F0:1`, `F1:0` precedes `F1:1`, and the backward tasks run `B1:1` before `B1:0` and `B0:1` before `B0:0`. The last stage-1 forward task gates the first backward task. This is a GPipe-style fill-and-drain **dependency model**, not a trace from PyTorch.

For a hand-checkable one-microbatch job, use `F0 = 2 ms`, activation = `4 MiB`, `F1 = 3 ms`, `B1 = 5 ms`, gradient = `2 MiB`, `B0 = 7 ms`, link capacity = `1 MiB/ms`, and zero delivery delay. There is no possible overlap along that single dependency chain, so completion is `2 + 4 + 3 + 5 + 2 + 7 = 23 ms`. [`test_sim_engine.py`](../tests/test_sim_engine.py) checks this case. With multiple microbatches or jobs, compute and transfers overlap and the scheduler matters.

Time is milliseconds, flow size is MiB, and link capacity is MiB/ms. Transmitting a flow of size `x` alone takes `x / capacity` ms; a fixed delivery delay follows transmission. Fair sharing divides capacity equally among ready flows. The simulator advances to the next compute completion, flow transmission completion, or delivery event, then reselects link service. Priority policies may preempt a transfer at these events without cost. These are explicit modeling assumptions, not measurements of a particular cluster.

The event loop in [`engine.py`](../wantrain/sim/engine.py) does the following:

1. Process all events at the current time. A job root makes its first compute task ready; a completed compute task releases dependent tasks or flows; a delivered flow releases its receiving compute task.
2. Start ready compute tasks on free workers. Compute tasks are non-preemptive and use **actual** durations.
3. Ask the selected network policy for rates over currently ready flows. FIFO, SRPT, and timing priority give the whole link to one flow; fair sharing divides it equally.
4. Advance time to the next queued event or transmission completion, deducting `rate × elapsed time` from each served flow. A finished transmission becomes a delivery event after the configured delay.
5. Reconsider the link whenever contenders or completions change. Save the selected flows and their comparison keys in the decision trace.
6. Stop when every task in every job has finished, then calculate per-job and flow metrics.

The event heap makes event order deterministic; flow IDs break policy ties. The current workload DAG fixes each worker's compute order, so the link is the only scheduling choice in this milestone.

## What each policy asks

The [scheduling policy reference](scheduling_policies.md) documents each rule, tie-break, input, and trace key in detail.

| Policy | Rule when flows contend | Question it tests |
|---|---|---|
| `fifo` | Earliest release first | Does arrival order suffice? |
| `fair` | Split capacity equally | What happens without a flow priority? |
| `srpt` | Smallest remaining transfer first | Is flow size alone a useful signal? |
| `echelon_static` | Earliest arrangement-time target first, using profiled compute | Does the receiving stage's compute rhythm help? |
| `echelon_oracle` | Same heuristic using actual compute durations | How much does profile accuracy matter? This is **not** a globally optimal schedule. |

EchelonFlow calls a related group of flows with staggered desired finish times an *echelon*. Here, each job's activation sequence and gradient sequence is one such group. The first flow sets an origin at its release time. For flow sequence number `m`, the target is `origin + m × estimated receiving-stage compute time`. Activations use `S1` forward time; gradients use `S0` backward time. The `echelon_static` policy reads the `profile_*` estimate for this target; `echelon_oracle` reads the actual duration as an **information control**. Both variants run the same actual workload. The heuristic gives the full link to the ready flow with the earliest target, breaking ties by remaining bytes and flow ID. It scans all ready flows at each event, taking `O(F)` selection work for `F` ready flows. This implementation does **not** optimize EchelonFlow's sum-of-maximum-tardiness objective or implement its proposed multi-link system.

## Change the information, keep the workload

```bash
python3 -m wantrain.sim.run --config experiments/configs/scheduling_profile_error_demo.yaml
```

This example keeps actual task durations and flow sizes identical. Job B's visible receiving-stage estimates are doubled. Consequently, `fifo`, `fair`, `srpt`, and `echelon_oracle` should give the same outcomes as the first example; `echelon_static` may change its decisions. Compare its timeline and per-job completion. This is a **crafted teaching case**, not evidence that the heuristic usually performs this way. The next experiment should vary contention and profile error systematically, using paired scenarios and reporting all outcomes.

The current model omits packet behavior, transport, topology, collectives, GPU overlap, optimizer work, and scheduler overhead. Its receiver-wait metric sums dependency wait across receiving stages and microbatches. That includes upstream computation and may exceed one job's elapsed step time; it must not be called pure network stall. Before using the artifact for outreach, we need a parameter sweep, an interpretable figure, and a trace-based explanation of the observed boundary.
