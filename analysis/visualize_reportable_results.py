#!/usr/bin/env python3
"""Analyze and visualize the reportable sentence-steering results.

The unit of analysis is one (base-trace, sentence-position) target. Generated
continuations are first averaged within target and condition; uncertainty is
then computed by paired bootstrap resampling of targets. This avoids treating
the 4 continuations from each of 5 replacement sentences as independent.

The script intentionally uses the LLM-only verdict for S/E/S-random because
the released Thought Branches ``contains_blackmail`` label is best reproduced
by that definition. I0 is loaded from ``chunk_<i+1>/solutions.json``, where the
original sentence i is held fixed and the continuation is resampled.

Outputs:
  * figure_s_vs_i0.png
  * figure_s_vs_controls.png
  * target_level_estimates.csv
  * summary.json
  * analysis.md

Only Pillow is required for plotting. The statistical analysis itself uses the
Python standard library.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import textwrap
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image, ImageDraw, ImageFont


REPORTABLE_S = "phase2_S_a15_classified.json"
REPORTABLE_CONTROLS = "phase2_E_Srand_classified.json"
THOUGHT_BRANCHES_SUBDIR = Path(
    "data/blackmail/qwq-32b/temperature_0.7_top_p_0.95/yes_base_solution"
)
TERMINAL_MARKER = "</tool_use:email>"

COLORS = {
    "ink": "#17212B",
    "muted": "#5E6B76",
    "grid": "#D9E0E6",
    "light_grid": "#EDF1F4",
    "paper": "#FFFFFF",
    "panel": "#F8FAFB",
    "S": "#D55E00",
    "I0": "#3B6FB6",
    "S-rand": "#009E73",
    "E": "#8B5CF6",
    "line": "#9BA7B2",
}


@dataclass(frozen=True, order=True)
class Target:
    scenario_id: str
    chunk_idx: int

    @property
    def label(self) -> str:
        return f"{self.scenario_id}/{self.chunk_idx}"


@dataclass
class TargetEstimate:
    target: Target
    rates_all: dict[str, float]
    rates_complete: dict[str, float]
    n_all: dict[str, int]
    n_complete: dict[str, int]


def mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("mean of empty sequence")
    return sum(values) / len(values)


def pct(value: float, digits: int = 1) -> str:
    return f"{100 * value:.{digits}f}%"


def pp(value: float, digits: int = 1, plus: bool = True) -> str:
    sign = "+" if plus else ""
    return f"{100 * value:{sign}.{digits}f} pp"


def percentile(sorted_values: Sequence[float], q: float) -> float:
    """Linearly interpolate a percentile from an already sorted sequence."""
    if not sorted_values:
        raise ValueError("percentile of empty sequence")
    position = (len(sorted_values) - 1) * q
    lo = math.floor(position)
    hi = math.ceil(position)
    if lo == hi:
        return sorted_values[lo]
    weight = position - lo
    return sorted_values[lo] * (1 - weight) + sorted_values[hi] * weight


def bootstrap_mean_ci(
    values: Sequence[float], samples: int, seed: int
) -> tuple[float, float]:
    """Percentile CI from resampling target-level estimates with replacement."""
    rng = random.Random(seed)
    n = len(values)
    draws = [
        sum(values[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(samples)
    ]
    draws.sort()
    return percentile(draws, 0.025), percentile(draws, 0.975)


def exact_sign_test(differences: Sequence[float], tolerance: float = 1e-12) -> dict:
    """Two-sided exact binomial sign test, dropping exact ties."""
    nonzero = [d for d in differences if abs(d) > tolerance]
    positive = sum(d > 0 for d in nonzero)
    negative = len(nonzero) - positive
    k = min(positive, negative)
    p_value = min(
        1.0,
        2 * sum(math.comb(len(nonzero), j) for j in range(k + 1)) / 2 ** len(nonzero),
    )
    return {
        "positive": positive,
        "negative": negative,
        "ties": len(differences) - len(nonzero),
        "p_value": p_value,
    }


def exact_sign_flip_test(differences: Sequence[float]) -> float:
    """Two-sided paired randomization test for the mean target-level effect."""
    observed = abs(mean(differences))
    n = len(differences)
    if n > 22:
        raise ValueError("Exact sign-flip enumeration is limited to 22 pairs")
    extreme = 0
    total = 1 << n
    for mask in range(total):
        value = 0.0
        for i, difference in enumerate(differences):
            value += difference if mask & (1 << i) else -difference
        if abs(value / n) >= observed - 1e-12:
            extreme += 1
    return extreme / total


def load_json(path: Path):
    with path.open() as handle:
        return json.load(handle)


def generated_values(rows: list[dict], condition: str, target: Target) -> tuple[list[int], list[int]]:
    all_values: list[int] = []
    complete_values: list[int] = []
    matching_rows = [
        row
        for row in rows
        if row.get("condition") == condition
        and str(row.get("scenario_id")) == target.scenario_id
        and int(row.get("chunk_idx")) == target.chunk_idx
    ]
    if len(matching_rows) != 5:
        raise ValueError(
            f"Expected 5 {condition} bursts for target {target.label}, got {len(matching_rows)}"
        )
    for row in matching_rows:
        continuations = row.get("continuations", [])
        verdicts = row.get("llm_verdict", [])
        if len(continuations) != len(verdicts):
            raise ValueError(f"Continuation/verdict length mismatch at {target.label}")
        for continuation, verdict in zip(continuations, verdicts):
            if not isinstance(verdict, bool):
                raise ValueError(f"Missing LLM verdict at {condition} {target.label}")
            all_values.append(int(verdict))
            if TERMINAL_MARKER in continuation:
                complete_values.append(int(verdict))
    if len(all_values) != 20:
        raise ValueError(
            f"Expected 20 {condition} continuations for target {target.label}, got {len(all_values)}"
        )
    if not complete_values:
        raise ValueError(f"No terminal-complete {condition} continuations at {target.label}")
    return all_values, complete_values


def i0_values(data_root: Path, target: Target) -> tuple[list[int], list[int]]:
    path = (
        data_root
        / THOUGHT_BRANCHES_SUBDIR
        / f"scenario_{target.scenario_id}"
        / f"chunk_{target.chunk_idx + 1}"
        / "solutions.json"
    )
    rows = load_json(path)
    all_values: list[int] = []
    complete_values: list[int] = []
    for row in rows:
        # Some released files append failed API attempts. They have no verdict and
        # are not part of the 100 successfully judged resamples.
        verdict = row.get("contains_blackmail")
        if not isinstance(verdict, bool):
            continue
        all_values.append(int(verdict))
        if TERMINAL_MARKER in (row.get("rollout") or ""):
            complete_values.append(int(verdict))
    if len(all_values) != 100:
        raise ValueError(f"Expected 100 valid I0 resamples at {target.label}, got {len(all_values)}")
    if not complete_values:
        raise ValueError(f"No terminal-complete I0 resamples at {target.label}")
    return all_values, complete_values


def target_set(rows: Iterable[dict], condition: str) -> set[Target]:
    return {
        Target(str(row["scenario_id"]), int(row["chunk_idx"]))
        for row in rows
        if row.get("condition") == condition
    }


def build_target_estimates(data_root: Path) -> tuple[list[TargetEstimate], dict]:
    reportable = data_root / "results" / "reportable_results"
    s_rows = load_json(reportable / REPORTABLE_S)
    control_rows = load_json(reportable / REPORTABLE_CONTROLS)
    rows = s_rows + control_rows

    condition_targets = {
        condition: target_set(rows, condition)
        for condition in ("S", "S-rand", "E")
    }
    common = set.intersection(*condition_targets.values())
    if not common:
        raise ValueError("No shared targets across S, S-rand, and E")

    excluded = {
        condition: sorted(condition_targets[condition] - common)
        for condition in condition_targets
    }
    targets = sorted(common, key=lambda t: (int(t.scenario_id), t.chunk_idx))
    estimates: list[TargetEstimate] = []
    for target in targets:
        rates_all: dict[str, float] = {}
        rates_complete: dict[str, float] = {}
        n_all: dict[str, int] = {}
        n_complete: dict[str, int] = {}

        i0_all, i0_complete = i0_values(data_root, target)
        rates_all["I0"] = mean(i0_all)
        rates_complete["I0"] = mean(i0_complete)
        n_all["I0"] = len(i0_all)
        n_complete["I0"] = len(i0_complete)

        for condition in ("S", "S-rand", "E"):
            values_all, values_complete = generated_values(rows, condition, target)
            rates_all[condition] = mean(values_all)
            rates_complete[condition] = mean(values_complete)
            n_all[condition] = len(values_all)
            n_complete[condition] = len(values_complete)

        estimates.append(
            TargetEstimate(target, rates_all, rates_complete, n_all, n_complete)
        )

    metadata = {
        "source_files": [
            str(reportable / REPORTABLE_S),
            str(reportable / REPORTABLE_CONTROLS),
        ],
        "condition_target_counts": {
            condition: len(targets_) for condition, targets_ in condition_targets.items()
        },
        "common_target_count": len(common),
        "common_base_trace_count": len({target.scenario_id for target in common}),
        "excluded_from_paired_comparisons": {
            condition: [target.label for target in targets_]
            for condition, targets_ in excluded.items()
        },
    }
    return estimates, metadata


def summarize_comparison(
    estimates: Sequence[TargetEstimate],
    comparator: str,
    subset: str,
    bootstrap_samples: int,
    seed: int,
) -> dict:
    rates_attr = "rates_complete" if subset == "terminal_complete" else "rates_all"
    s_rates = [getattr(estimate, rates_attr)["S"] for estimate in estimates]
    comparator_rates = [getattr(estimate, rates_attr)[comparator] for estimate in estimates]
    differences = [s - other for s, other in zip(s_rates, comparator_rates)]
    ci_low, ci_high = bootstrap_mean_ci(differences, bootstrap_samples, seed)
    return {
        "comparison": f"S - {comparator}",
        "subset": subset,
        "targets": len(estimates),
        "s_mean_target_rate": mean(s_rates),
        "comparator_mean_target_rate": mean(comparator_rates),
        "mean_paired_difference": mean(differences),
        "paired_bootstrap_95_ci": [ci_low, ci_high],
        "exact_sign_test": exact_sign_test(differences),
        "exact_mean_sign_flip_p": exact_sign_flip_test(differences),
        "target_differences": {
            estimate.target.label: difference
            for estimate, difference in zip(estimates, differences)
        },
    }


def build_summary(
    estimates: Sequence[TargetEstimate], metadata: dict, bootstrap_samples: int, seed: int
) -> dict:
    arms = {}
    for condition in ("I0", "S", "S-rand", "E"):
        for subset, attr in (("all", "rates_all"), ("terminal_complete", "rates_complete")):
            rates = [getattr(estimate, attr)[condition] for estimate in estimates]
            counts = [
                getattr(estimate, "n_complete" if subset == "terminal_complete" else "n_all")[condition]
                for estimate in estimates
            ]
            ci = bootstrap_mean_ci(rates, bootstrap_samples, seed + len(arms) * 17)
            arms[f"{condition}_{subset}"] = {
                "mean_target_rate": mean(rates),
                "target_bootstrap_95_ci": list(ci),
                "positive_continuations": sum(
                    round(rate * count) for rate, count in zip(rates, counts)
                ),
                "continuations": sum(counts),
            }

    comparisons = {}
    comparison_specs = [
        ("s_vs_i0_all_observed", "I0", "all"),
        ("s_vs_i0_terminal_complete", "I0", "terminal_complete"),
        ("s_vs_srand_all", "S-rand", "all"),
        ("s_vs_srand_terminal_complete", "S-rand", "terminal_complete"),
        ("s_vs_e_all", "E", "all"),
        ("s_vs_e_terminal_complete", "E", "terminal_complete"),
    ]
    for index, (name, comparator, subset) in enumerate(comparison_specs):
        comparisons[name] = summarize_comparison(
            estimates,
            comparator,
            subset,
            bootstrap_samples,
            seed + 1000 + index * 101,
        )

    return {
        "analysis_unit": "base-trace/sentence-position target",
        "outcome": "LLM-only blackmail verdict",
        "terminal_marker": TERMINAL_MARKER,
        "bootstrap_samples": bootstrap_samples,
        "random_seed": seed,
        "metadata": metadata,
        "arms": arms,
        "comparisons": comparisons,
    }


def write_target_csv(path: Path, estimates: Sequence[TargetEstimate]) -> None:
    fieldnames = ["scenario_id", "chunk_idx"]
    for condition in ("I0", "S", "S-rand", "E"):
        fieldnames.extend(
            [
                f"{condition}_n_all",
                f"{condition}_rate_all",
                f"{condition}_n_terminal_complete",
                f"{condition}_rate_terminal_complete",
            ]
        )
    fieldnames.extend(
        [
            "S_minus_I0_terminal_complete",
            "S_minus_S-rand_all",
            "S_minus_E_all",
        ]
    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for estimate in estimates:
            row = {
                "scenario_id": estimate.target.scenario_id,
                "chunk_idx": estimate.target.chunk_idx,
            }
            for condition in ("I0", "S", "S-rand", "E"):
                row.update(
                    {
                        f"{condition}_n_all": estimate.n_all[condition],
                        f"{condition}_rate_all": estimate.rates_all[condition],
                        f"{condition}_n_terminal_complete": estimate.n_complete[condition],
                        f"{condition}_rate_terminal_complete": estimate.rates_complete[condition],
                    }
                )
            row.update(
                {
                    "S_minus_I0_terminal_complete": estimate.rates_complete["S"]
                    - estimate.rates_complete["I0"],
                    "S_minus_S-rand_all": estimate.rates_all["S"]
                    - estimate.rates_all["S-rand"],
                    "S_minus_E_all": estimate.rates_all["S"] - estimate.rates_all["E"],
                }
            )
            writer.writerow(row)


class Figure:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.image = Image.new("RGB", (width, height), COLORS["paper"])
        self.draw = ImageDraw.Draw(self.image)
        self.fonts = self._load_fonts()

    @staticmethod
    def _load_fonts() -> dict[tuple[str, int], ImageFont.FreeTypeFont | ImageFont.ImageFont]:
        paths = {
            "regular": Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
            "bold": Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
            "italic": Path("/System/Library/Fonts/Supplemental/Arial Italic.ttf"),
        }
        fonts = {}
        for style, path in paths.items():
            for size in (24, 25, 26, 28, 30, 32, 34, 38, 42, 50, 58):
                try:
                    fonts[(style, size)] = ImageFont.truetype(str(path), size)
                except OSError:
                    fonts[(style, size)] = ImageFont.load_default()
        return fonts

    def font(self, size: int, style: str = "regular"):
        return self.fonts[(style, size)]

    def text(
        self,
        xy: tuple[float, float],
        value: str,
        size: int = 28,
        color: str = COLORS["ink"],
        style: str = "regular",
        anchor: str = "la",
    ) -> None:
        self.draw.text(xy, value, font=self.font(size, style), fill=color, anchor=anchor)

    def wrapped_text(
        self,
        xy: tuple[int, int],
        value: str,
        width_chars: int,
        size: int = 26,
        color: str = COLORS["muted"],
        style: str = "regular",
        spacing: int = 8,
    ) -> None:
        wrapped = "\n".join(textwrap.wrap(value, width=width_chars))
        self.draw.multiline_text(
            xy,
            wrapped,
            font=self.font(size, style),
            fill=color,
            spacing=spacing,
        )

    def save(self, path: Path) -> None:
        self.image.save(path, dpi=(200, 200), optimize=True)


def draw_axes(
    fig: Figure,
    bounds: tuple[int, int, int, int],
    x_label: str,
    y_label: str,
    panel_label: str,
    panel_title: str,
) -> tuple:
    x0, y0, x1, y1 = bounds
    fig.draw.rounded_rectangle(bounds, radius=10, fill=COLORS["panel"], outline=COLORS["grid"], width=2)
    fig.text((x0 + 24, y0 + 20), panel_label, 30, style="bold")
    fig.text((x0 + 75, y0 + 20), panel_title, 30, style="bold")

    plot = (x0 + 125, y0 + 110, x1 - 40, y1 - 105)
    px0, py0, px1, py1 = plot
    for tick in range(0, 6):
        value = tick / 5
        x = px0 + value * (px1 - px0)
        y = py1 - value * (py1 - py0)
        fig.draw.line((x, py0, x, py1), fill=COLORS["light_grid"], width=2)
        fig.draw.line((px0, y, px1, y), fill=COLORS["light_grid"], width=2)
        fig.text((x, py1 + 18), f"{int(value * 100)}%", 24, COLORS["muted"], anchor="ma")
        fig.text((px0 - 20, y), f"{int(value * 100)}%", 24, COLORS["muted"], anchor="rm")
    fig.draw.rectangle(plot, outline=COLORS["grid"], width=2)
    fig.text(((px0 + px1) / 2, y1 - 38), x_label, 26, COLORS["ink"], anchor="ma")
    # Pillow supports rotated text most reliably through a transparent layer.
    label_box = Image.new("RGBA", (450, 50), (0, 0, 0, 0))
    label_draw = ImageDraw.Draw(label_box)
    label_draw.text((225, 25), y_label, font=fig.font(26), fill=COLORS["ink"], anchor="mm")
    label_box = label_box.rotate(90, expand=True)
    fig.image.paste(label_box, (x0 + 5, int((py0 + py1) / 2 - label_box.height / 2)), label_box)
    return plot


def map_point(plot: tuple[int, int, int, int], x: float, y: float) -> tuple[float, float]:
    x0, y0, x1, y1 = plot
    return x0 + x * (x1 - x0), y1 - y * (y1 - y0)


def draw_identity(fig: Figure, plot: tuple[int, int, int, int]) -> None:
    x0, y0 = map_point(plot, 0, 0)
    x1, y1 = map_point(plot, 1, 1)
    segments = 18
    for i in range(segments):
        if i % 2 == 0:
            xa = x0 + (x1 - x0) * i / segments
            ya = y0 + (y1 - y0) * i / segments
            xb = x0 + (x1 - x0) * (i + 1) / segments
            yb = y0 + (y1 - y0) * (i + 1) / segments
            fig.draw.line((xa, ya, xb, yb), fill=COLORS["line"], width=3)
    fig.text((x1 - 8, y1 + 22), "equal rates", 24, COLORS["muted"], anchor="ra")


def draw_scatter(
    fig: Figure,
    plot: tuple[int, int, int, int],
    x_values: Sequence[float],
    y_values: Sequence[float],
    point_color: str,
    labels: Sequence[str],
    label_extremes: int = 4,
) -> None:
    differences = [y - x for x, y in zip(x_values, y_values)]
    label_indices = set(
        sorted(range(len(differences)), key=lambda i: abs(differences[i]), reverse=True)[:label_extremes]
    )
    for index, (x_value, y_value) in enumerate(zip(x_values, y_values)):
        x, y = map_point(plot, x_value, y_value)
        fig.draw.ellipse((x - 10, y - 10, x + 10, y + 10), fill=point_color, outline="white", width=3)
        if index in label_indices:
            anchor = "la" if x < (plot[0] + plot[2]) / 2 else "ra"
            dx = 14 if anchor == "la" else -14
            fig.text((x + dx, y - 4), labels[index], 24, COLORS["ink"], anchor=anchor)


def add_summary_block(
    fig: Figure,
    xy: tuple[int, int],
    title: str,
    lines: Sequence[str],
    accent: str,
) -> None:
    x, y = xy
    fig.draw.line((x, y, x + 8, y + 128), fill=accent, width=8)
    fig.text((x + 30, y - 4), title, 28, style="bold")
    for index, line in enumerate(lines):
        fig.text((x + 30, y + 40 + index * 36), line, 26, COLORS["muted"])


def figure_s_vs_i0(
    path: Path, estimates: Sequence[TargetEstimate], summary: dict
) -> None:
    fig = Figure(2000, 1280)
    fig.text((80, 55), "S versus I0: lower blackmail", 50, style="bold")
    fig.text(
        (80, 122),
        "QwQ-32B · layer 29 · α=1.5 · generated-token scope · 19 paired trace-position targets",
        28,
        COLORS["muted"],
    )

    plot = draw_axes(
        fig,
        (80, 190, 1330, 1010),
        "I0: original sentence retained",
        "S: steered replacement",
        "A",
        "Terminal-complete blackmail rate by target",
    )
    draw_identity(fig, plot)
    x_values = [estimate.rates_complete["I0"] for estimate in estimates]
    y_values = [estimate.rates_complete["S"] for estimate in estimates]
    draw_scatter(
        fig,
        plot,
        x_values,
        y_values,
        COLORS["S"],
        [estimate.target.label for estimate in estimates],
        label_extremes=0,
    )
    primary = summary["comparisons"]["s_vs_i0_terminal_complete"]
    ci_low, ci_high = primary["paired_bootstrap_95_ci"]
    add_summary_block(
        fig,
        (1400, 275),
        "Paired target estimate",
        [
            f"I0 {pct(primary['comparator_mean_target_rate'])}  →  S {pct(primary['s_mean_target_rate'])}",
            f"Δ = {pp(primary['mean_paired_difference'])}",
            f"95% target bootstrap CI [{pp(ci_low)}, {pp(ci_high)}]",
        ],
        COLORS["S"],
    )

    s_complete = summary["arms"]["S_terminal_complete"]
    i0_complete = summary["arms"]["I0_terminal_complete"]
    footnote = (
        f"Each point is one base-trace/sentence-position target. "
        f"S uses {s_complete['continuations']}/380 terminal-complete continuations; "
        f"I0 uses {i0_complete['continuations']}/1900."
    )
    fig.wrapped_text((85, 1060), footnote, 145, size=26, color=COLORS["muted"])
    fig.save(path)


def draw_control_panel(
    fig: Figure,
    bounds: tuple[int, int, int, int],
    estimates: Sequence[TargetEstimate],
    comparator: str,
    panel_label: str,
    summary: dict,
) -> None:
    plot = draw_axes(
        fig,
        bounds,
        f"{comparator}: comparator blackmail rate",
        "S: steered blackmail rate",
        panel_label,
        f"S against {comparator}",
    )
    draw_identity(fig, plot)
    x_values = [estimate.rates_all[comparator] for estimate in estimates]
    y_values = [estimate.rates_all["S"] for estimate in estimates]
    draw_scatter(
        fig,
        plot,
        x_values,
        y_values,
        COLORS[comparator],
        [estimate.target.label for estimate in estimates],
        label_extremes=0,
    )
    ci_low, ci_high = summary["paired_bootstrap_95_ci"]
    summary_y = bounds[3] + 38
    fig.text(
        (bounds[0] + 25, summary_y),
        f"Means: {comparator} {pct(summary['comparator_mean_target_rate'])}, S {pct(summary['s_mean_target_rate'])}",
        26,
        style="bold",
    )
    fig.text(
        (bounds[0] + 25, summary_y + 42),
        f"Paired Δ {pp(summary['mean_paired_difference'])}  ·  95% CI [{pp(ci_low)}, {pp(ci_high)}]  ·  sign p={summary['exact_sign_test']['p_value']:.3f}",
        25,
        COLORS["muted"],
    )


def figure_s_vs_controls(
    path: Path, estimates: Sequence[TargetEstimate], summary: dict
) -> None:
    fig = Figure(2200, 1420)
    fig.text((80, 55), "Steering does not clearly separate from generated controls", 50, style="bold")
    fig.text(
        (80, 122),
        "All continuations · same 19 targets and 380 continuations per arm · LLM-only blackmail verdict",
        28,
        COLORS["muted"],
    )
    draw_control_panel(
        fig,
        (80, 190, 1070, 1100),
        estimates,
        "S-rand",
        "B",
        summary["comparisons"]["s_vs_srand_all"],
    )
    draw_control_panel(
        fig,
        (1130, 190, 2120, 1100),
        estimates,
        "E",
        "C",
        summary["comparisons"]["s_vs_e_all"],
    )

    completion = {
        condition: summary["arms"][f"{condition}_terminal_complete"]["continuations"] / 380
        for condition in ("S", "S-rand", "E")
    }
    footnote = (
        "Each point is one paired trace-position target. "
        f"Terminal-completion rates were similar across the generated arms "
        f"(S {pct(completion['S'])}, S-rand {pct(completion['S-rand'])}, E {pct(completion['E'])}), "
        "so the stop-string bug is unlikely to manufacture either within-generated contrast. "
        "Intervals are paired percentile bootstraps over targets; exact sign tests use only the direction of each target effect."
    )
    fig.wrapped_text((85, 1255), footnote, 164, size=26, color=COLORS["muted"])
    fig.save(path)


def write_analysis(path: Path, summary: dict) -> None:
    i0 = summary["comparisons"]["s_vs_i0_terminal_complete"]
    i0_all = summary["comparisons"]["s_vs_i0_all_observed"]
    srand = summary["comparisons"]["s_vs_srand_all"]
    e = summary["comparisons"]["s_vs_e_all"]
    i0_ci = i0["paired_bootstrap_95_ci"]
    srand_ci = srand["paired_bootstrap_95_ci"]
    e_ci = e["paired_bootstrap_95_ci"]
    exclusions = summary["metadata"]["excluded_from_paired_comparisons"]

    text = f"""# Reportable result analysis: S vs I0, S-random, and E

## Analysis contract

- **Unit:** one `(scenario_id, chunk_idx)` base-trace/sentence-position target, not one continuation. Here `scenario_id` names a sampled base reasoning trace, not a distinct scenario.
- **Paired sample:** {summary['metadata']['common_target_count']} targets from {summary['metadata']['common_base_trace_count']} independently sampled reasoning traces for one blackmail scenario. The E/S-random file has 21 targets, but S is missing {', '.join(exclusions['E'])}; all S comparisons use the 19-target intersection.
- **Outcome:** LLM-only blackmail verdict. This matches the released Thought Branches `contains_blackmail` labels better than the local compound email/keyword/LLM gate.
- **I0 construction:** for target sentence `i`, use the 100 valid continuations in `chunk_(i+1)/solutions.json`, where sentence `i` remains in the prefix.
- **Aggregation:** average continuations within each target and arm, then compute paired target differences. 95% intervals are percentile bootstraps over targets ({summary['bootstrap_samples']:,} samples, seed {summary['random_seed']}).

## Finding 1: S against I0

Among terminal-complete continuations, mean target-level blackmail fell from **{pct(i0['comparator_mean_target_rate'])} under I0 to {pct(i0['s_mean_target_rate'])} under S**, a paired difference of **{pp(i0['mean_paired_difference'])}** (95% target-bootstrap CI **[{pp(i0_ci[0])}, {pp(i0_ci[1])}]**). S was lower at {i0['exact_sign_test']['negative']} of {i0['targets']} targets and higher at {i0['exact_sign_test']['positive']}; the exact sign test is **p={i0['exact_sign_test']['p_value']:.3f}**, while the exact sign-flip test for the mean magnitude is **p={i0['exact_mean_sign_flip_p']:.3f}**. The magnitude-weighted result is directional, but the target directions are heterogeneous.

The unfiltered observed contrast is larger (**{pp(i0_all['mean_paired_difference'])}**), but it is not a valid headline estimate: a stop string ended about one-third of generated continuations immediately before the action, whereas I0 nearly always ran to completion. Restricting to terminal-complete outputs reduces that mismatch, but completion is post-treatment, so the {pp(i0['mean_paired_difference'])} estimate remains a sensitivity analysis rather than a clean causal effect.

## Finding 2: S against S-random and E

Across the same targets and using all 20 generated continuations per target and arm, S averaged **{pct(srand['s_mean_target_rate'])}** blackmail versus **{pct(srand['comparator_mean_target_rate'])}** for S-random: **{pp(srand['mean_paired_difference'])}** (95% CI **[{pp(srand_ci[0])}, {pp(srand_ci[1])}]**, sign-test p={srand['exact_sign_test']['p_value']:.3f}). This is a small, uncertain difference, so the reportable results do **not** show a clear separation between the reconsideration vector and norm-matched random steering.

S also closely matches the direct-edit arm E: **{pct(e['s_mean_target_rate'])}** for S versus **{pct(e['comparator_mean_target_rate'])}** for E, a paired difference of **{pp(e['mean_paired_difference'])}** (95% CI **[{pp(e_ci[0])}, {pp(e_ci[1])}]**, sign-test p={e['exact_sign_test']['p_value']:.3f}). On the measured outcome, S and E are effectively indistinguishable at this sample size.

## Interpretation and limitations

The strongest defensible story is therefore asymmetric: S is associated with less blackmail than the original-sentence I0 baseline, but it does not outperform the two generated controls. That weakens a mechanism-specific interpretation. The S-random control is especially important: because it receives the same intervention machinery without the reconsideration direction, the small S-minus-S-random contrast is consistent with much of the S/I0 shift coming from generic generation/intervention effects rather than reconsideration content alone.

This conclusion is limited by (1) the stop-string truncation mismatch against I0, (2) conditioning on terminal completion in the sensitivity estimate, (3) only 19 targets, with multiple targets from some base traces, (4) the two missing classified S targets, (5) one scenario, model, task, layer, and coefficient, and (6) E's style confound. The figure should be paired with randomly sampled qualitative examples and a manual audit of judge labels in the final write-up.
"""
    path.write_text(text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root containing results/ and data/ (default: parent of analysis/)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "output",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20260904)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    estimates, metadata = build_target_estimates(args.data_root)
    summary = build_summary(estimates, metadata, args.bootstrap_samples, args.seed)

    write_target_csv(args.output_dir / "target_level_estimates.csv", estimates)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_analysis(args.output_dir / "analysis.md", summary)
    figure_s_vs_i0(args.output_dir / "figure_s_vs_i0.png", estimates, summary)
    figure_s_vs_controls(args.output_dir / "figure_s_vs_controls.png", estimates, summary)

    print(f"Analyzed {len(estimates)} paired targets.")
    for name in (
        "s_vs_i0_terminal_complete",
        "s_vs_srand_all",
        "s_vs_e_all",
    ):
        result = summary["comparisons"][name]
        ci_low, ci_high = result["paired_bootstrap_95_ci"]
        print(
            f"{result['comparison']} ({result['subset']}): "
            f"{pp(result['mean_paired_difference'])} "
            f"[{pp(ci_low)}, {pp(ci_high)}]"
        )
    print(f"Wrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
