"""Render exact simulator intervals as a time-scaled SVG timeline."""

from __future__ import annotations

from html import escape
from math import ceil

from wantrain.sim.engine import Result
from wantrain.sim.workload import Scenario


def plot_svg(results: list[Result], scenario: Scenario) -> str:
    """Show compute and link service on a common millisecond axis."""
    horizon = max(end for result in results for end in result.task_end_ms.values())
    plot_left, plot_width = 195, max(900, ceil(horizon * 12))
    row_height, panel_gap, header_height = 30, 30, 101
    rows_per_panel = len(scenario.jobs) * 4
    panel_height = 42 + rows_per_panel * row_height + panel_gap
    width = plot_left + plot_width + 35
    height = header_height + len(results) * panel_height + 20
    colors = {
        "forward": "#2563eb",
        "backward": "#7c3aed",
        "activation": "#059669",
        "gradient": "#ea580c",
    }

    def x(time_ms: float) -> float:
        return plot_left + time_ms / horizon * plot_width

    def flow_runs(result: Result) -> list[tuple[str, float, float]]:
        """Join event-split intervals while a flow keeps receiving service."""
        runs: dict[str, list[tuple[float, float]]] = {}
        for interval in result.network_intervals:
            for task in interval.tasks:
                segments = runs.setdefault(task, [])
                if segments and abs(segments[-1][1] - interval.start_ms) < 1e-9:
                    segments[-1] = (segments[-1][0], interval.end_ms)
                else:
                    segments.append((interval.start_ms, interval.end_ms))
        return [(task, start, end) for task, segments in runs.items() for start, end in segments]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" '
        'aria-label="Pipeline compute and shared link timeline">',
        '<style>text{font-family:system-ui,Arial,sans-serif;fill:#1f2937}'
        '.small{font-size:12px}.label{font-size:13px}.title{font-size:17px;font-weight:700}'
        '.bartext{font-size:11px;fill:white;font-weight:700;pointer-events:none}'
        '.grid{stroke:#d1d5db;stroke-width:1}.rowline{stroke:#edf0f3;stroke-width:1}'
        '</style>',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        '<text x="20" y="26" class="title">Pipeline timeline</text>',
        '<text x="20" y="47" class="small">Bar width shows elapsed time. '
        'Transfer bars show shared link service; gaps are waiting.</text>',
    ]
    for index, (phase, color) in enumerate(colors.items()):
        legend_x = 20 + index * 165
        parts.append(f'<rect x="{legend_x}" y="59" width="16" height="16" '
                     f'fill="{color}" rx="2"/>')
        parts.append(f'<text x="{legend_x + 23}" y="72" class="small">'
                     f'{phase.title()}</text>')

    tick_step = 10 if horizon <= 120 else (20 if horizon <= 240 else 50)
    for panel_index, result in enumerate(results):
        top = header_height + panel_index * panel_height
        first_row_y = top + 42
        bottom = first_row_y + rows_per_panel * row_height
        completions = "  ·  ".join(
            f"{job.name} step {result.job_completion_ms[job.name]:g} ms"
            for job in scenario.jobs
        )
        parts.append(f'<text x="20" y="{top + 17}" class="title">'
                     f'{escape(result.policy)}  ·  {escape(completions)}</text>')
        for tick in range(0, ceil(horizon / tick_step) * tick_step + 1, tick_step):
            if tick > horizon:
                break
            px = x(tick)
            parts.append(f'<line x1="{px:.2f}" y1="{top + 27}" x2="{px:.2f}" '
                         f'y2="{bottom}" class="grid"/>')
            parts.append(f'<text x="{px:.2f}" y="{top + 36}" class="small" '
                         f'text-anchor="middle">{tick}</text>')

        service_runs = flow_runs(result)
        for job_index, job in enumerate(scenario.jobs):
            base = first_row_y + job_index * 4 * row_height
            labels = ("stage 0 compute", "stage 1 compute", "activation · link", "gradient · link")
            for row_index, label in enumerate(labels):
                y = base + row_index * row_height
                parts.append(f'<text x="20" y="{y + 18}" class="label">'
                             f'{escape(job.name)}  {label}</text>')
                parts.append(f'<line x1="{plot_left}" y1="{y + row_height - 1}" '
                             f'x2="{x(horizon):.2f}" y2="{y + row_height - 1}" class="rowline"/>')

            for interval in result.compute_intervals:
                task = interval.tasks[0]
                name, kind, microbatch = task.split(":")
                if name != job.name:
                    continue
                row_index = 0 if kind.endswith("0") else 1
                y = base + row_index * row_height + 5
                left, bar_width = x(interval.start_ms), x(interval.end_ms) - x(interval.start_ms)
                phase = "forward" if kind.startswith("F") else "backward"
                title = f"{job.name} stage {row_index} {phase} m{microbatch}: {interval.start_ms:g}–{interval.end_ms:g} ms"
                parts.append(f'<rect x="{left:.2f}" y="{y}" width="{bar_width:.2f}" height="20" '
                             f'fill="{colors[phase]}" stroke="white" stroke-width="1" rx="2">'
                             f'<title>{escape(title)}</title></rect>')
                if bar_width >= 40:
                    parts.append(f'<text x="{left + 4:.2f}" y="{y + 14}" class="bartext">'
                                 f'{phase[0].upper()} m{microbatch}</text>')

            for kind, flow, row_index in (("A", "activation", 2), ("G", "gradient", 3)):
                matching = sorted(
                    [(task, start, end) for task, start, end in service_runs
                     if task.startswith(f"{job.name}:{kind}:")],
                    key=lambda run: run[1],
                )
                # Put simultaneous flows in separate slots on the same row.
                slotted: list[tuple[str, float, float, int]] = []
                slot_ends: list[float] = []
                for task, start, end in matching:
                    slot = next((i for i, prior_end in enumerate(slot_ends)
                                 if prior_end <= start + 1e-9), len(slot_ends))
                    if slot == len(slot_ends):
                        slot_ends.append(end)
                    else:
                        slot_ends[slot] = end
                    slotted.append((task, start, end, slot))
                bar_height = 20 / max(1, len(slot_ends))
                y = base + row_index * row_height + 5
                for task, start, end, slot in slotted:
                    microbatch = task.rsplit(":", 1)[1]
                    left, bar_width = x(start), x(end) - x(start)
                    size = job.activation_mib if kind == "A" else job.gradient_mib
                    title = (f"{job.name} {flow} m{microbatch} ({size:g} MiB): "
                             f"link service {start:g}–{end:g} ms")
                    parts.append(f'<rect x="{left:.2f}" y="{y + slot * bar_height:.2f}" '
                                 f'width="{bar_width:.2f}" height="{bar_height:.2f}" '
                                 f'fill="{colors[flow]}" stroke="white" stroke-width="1" rx="2">'
                                 f'<title>{escape(title)}</title></rect>')
                    if bar_width >= 45 and bar_height >= 13:
                        parts.append(f'<text x="{left + 4:.2f}" y="{y + slot * bar_height + 14:.2f}" '
                                     f'class="bartext">{kind} m{microbatch}</text>')

    parts.append('</svg>')
    return "\n".join(parts) + "\n"
