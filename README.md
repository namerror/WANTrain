# WANTrain

WANTrain explores how distributed training should use a network when compute and communication do not progress at uniform speeds. The **current experiment** asks a narrow question: when two pipeline-training jobs share a bottleneck link, how does the order of their transfers change worker idle time and step completion?

The repository currently has a **synthetic, discrete-event flow-scheduling simulator** and a separate **single-process PyTorch training reference**. The simulator is the active research milestone. It does not train a model, run a distributed PyTorch job, or emulate packets. The PyTorch reference can later help calibrate compute times and tensor sizes; it is not connected to the simulator yet.

| Part | Status | Purpose |
|---|---|---|
| Pipeline flow simulator | Runs now | Compare link policies on the same two-stage training workload and inspect their decisions. |
| Local PyTorch reference | Runs now | Train a small synthetic classifier or local CIFAR-10/ResNet-18 and record telemetry. |
| Multi-process training, WAN emulation, optical networks | Future work | Validate and extend questions raised by the controlled simulator. |

## Try the simulator

The simulator itself needs Python 3.11+ and PyYAML. From the repository root, [uv](https://docs.astral.sh/uv/) can install the **full** project environment, including PyTorch and test tools:

```bash
uv sync --extra dev
uv run python -m wantrain.sim.run --config experiments/configs/scheduling_demo.yaml
```

For a time-scaled plot of compute and transfers, run:

```bash
uv run python -m wantrain.sim.run --config experiments/configs/scheduling_demo.yaml \
  --policy fifo --plot temp/fifo.svg
```

Open `temp/fifo.svg` in a browser. Blue, purple, green, and orange mark forward compute, backward compute, activation transfers, and gradient transfers. Each bar has a tooltip with its microbatch and exact times. The activation rows show when the shared link serves each activation; a 6 MiB flow on an otherwise free 1 MiB/ms link spans 6 ms.

The command prints a table and a timeline for five policies: FIFO, equal bandwidth sharing, shortest remaining flow, and a timing-aware heuristic with either profiled or actual compute times. All policies see the **same actual work and link capacity**. To inspect why a policy chose a flow:

```bash
uv run python -m wantrain.sim.run --config experiments/configs/scheduling_demo.yaml \
  --policy fifo --policy srpt --policy echelon_static --decisions
```

At **3 ms**, Job B's 2 MiB activation transfer becomes ready while Job A's transfer has 5 MiB left. `srpt` switches to B; `fifo` and `echelon_static` continue A. The printed timeline lets you follow that choice into each job's stage-1 compute and eventual step completion. See the [policy reference](docs/scheduling_policies.md) for every policy's rule and tie-breaks, and the [guided simulator walkthrough](docs/flow_scheduling_design.md) for the dependency graph and trace notation.

You can run the module with `python3 -m wantrain.sim.run` instead if PyYAML is already installed in that Python environment. Tests:

```bash
uv run pytest -q
```

## What is being simulated?

Each of two jobs has two modeled workers. It splits one training step into microbatches. For each microbatch, stage 0 computes, an activation crosses the shared link, and stage 1 computes. After the forward phase, gradients flow back in reverse microbatch order:

```text
stage 0 forward → activation flow → stage 1 forward
stage 0 backward ← gradient flow  ← stage 1 backward
```

A worker computes one task at a time. A receiver waits until both its local predecessor and the required flow finish. The network has one shared capacity and a fixed delivery delay. Scheduling decides **which ready flow receives link capacity**; it does not change the amount of computation or data.

This isolates a useful systems question, but it leaves out GPU overlap, transport, packet loss, topology, collective communication, and training convergence. The timing-aware rule is **inspired by** [EchelonFlow's staggered desired flow completion times](https://conferences.sigcomm.org/hotnets/2022/papers/hotnets22_pan.pdf). It is our simple earliest-target heuristic, not an implementation or reproduction of EchelonFlow's full optimization or system.

## Experiments available today

| Config | What changes | What it teaches |
|---|---|---|
| [`scheduling_demo.yaml`](experiments/configs/scheduling_demo.yaml) | Two jobs with different compute times and flow sizes share a 1 MiB/ms link. | How FIFO, fair sharing, size priority, and timing priority create different schedules. |
| [`scheduling_profile_error_demo.yaml`](experiments/configs/scheduling_profile_error_demo.yaml) | Only Job B's **scheduler-visible** receiving-stage compute estimates double. Actual compute and bytes stay the same. | How a wrong profile can change a timing-aware choice without changing the underlying workload. |

These are **crafted teaching examples**, not benchmark evidence that one policy is generally better. The next research experiment will sweep link capacity, compute heterogeneity, and signed profile error on paired workloads, then explain the outcomes with both a figure and exact traces. The [experiment guide](docs/experiments.md) explains the config fields, metrics, comparison rules, and how to add a scenario.

## Repository map

| Path | Role |
|---|---|
| [`wantrain/sim/workload.py`](wantrain/sim/workload.py) | Parses simulator YAML and builds the forward/backward task dependency graph. |
| [`wantrain/sim/engine.py`](wantrain/sim/engine.py) | Advances compute and flow events, applies link policies, records decisions and metrics. |
| [`wantrain/sim/run.py`](wantrain/sim/run.py) | Runs policies, prints timelines, and optionally saves JSON traces and SVG plots. |
| [`experiments/configs/`](experiments/configs/) | Input scenarios for the simulator and local PyTorch reference. |
| [`tests/test_sim_engine.py`](tests/test_sim_engine.py) | Checks dependency order, link accounting, policy choices, determinism, and profile isolation. |
| [`wantrain/reference.py`](wantrain/reference.py), [`wantrain/workloads.py`](wantrain/workloads.py) | Separate single-process PyTorch reference path. |
| [`wantrain/config.py`](wantrain/config.py), [`wantrain/telemetry.py`](wantrain/telemetry.py), [`wantrain/cli.py`](wantrain/cli.py) | Configuration, event logging, and CLI for the reference path. |
| [`docs/`](docs/) | Model walkthrough, experiment guide, and a proposed later distributed runtime design. |

Generated runs and local work go under `runs/`, `data/`, and `temp/`; Git ignores those directories. The local learning checklist is `temp/progress.md` when present, so it does not appear in a fresh clone.

## Local PyTorch reference

This is a separate baseline for real model execution. It is **single-process**, with no simulated flow scheduler or distributed workers:

```bash
uv run wantrain reference run --config experiments/configs/smoke_cpu.yaml
```

The CPU smoke config uses a tiny synthetic classifier. [`cifar10_resnet18_local.yaml`](experiments/configs/cifar10_resnet18_local.yaml) is a longer local training config that may download CIFAR-10 and use CUDA if available. Reference runs write a resolved YAML config, JSONL events, and `metrics.json` under `runs/`. The [proposed runtime architecture](docs/architecture.md) records a later direction; its coordinator, transport, and synchronization policies are not implemented here.

## Read next

1. [Simulator walkthrough](docs/flow_scheduling_design.md): follow one microbatch and one scheduling decision through the code.
2. [Scheduling policy reference](docs/scheduling_policies.md): inspect each policy's inputs, allocation rule, tie-breaks, and trace key.
3. [Experiment guide](docs/experiments.md): change a scenario, compare policies fairly, and interpret outputs.
4. [Proposed distributed runtime architecture](docs/architecture.md): longer-term ideas, separate from the current simulator.

## License

No license has been selected yet.
