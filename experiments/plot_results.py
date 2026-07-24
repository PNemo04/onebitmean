#!/usr/bin/env python3
"""Create grayscale, publication-ready figures from saved CSV records."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FuncFormatter


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9.0,
        "legend.fontsize": 7.5,
        "figure.dpi": 180,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.35,
        "lines.markersize": 4.0,
    })


def plot_scaling(rows: list[dict[str, str]]) -> None:
    style()
    cases = [(3.0, r"$k=3$"), (2.0, r"$k=2$"), (1.5, r"$k=1.5$")]
    backend_style = {
        "dyadic": {"color": "#202020", "linestyle": "-", "marker": "o", "label": "dyadic"},
        "continuous": {"color": "#777777", "linestyle": "--", "marker": "s", "label": "continuous"},
    }
    fig, axes = plt.subplots(1, 3, figsize=(6.75, 2.05), sharey=False)
    for ax, (k, title) in zip(axes, cases):
        for backend in ("dyadic", "continuous"):
            subset = [row for row in rows if float(row["k"]) == k and row["backend"] == backend]
            subset.sort(key=lambda row: float(row["epsilon"]), reverse=True)
            x = np.array([float(row["epsilon"]) / float(row["tau"]) for row in subset])
            y = np.array([float(row["normalized_second_moment"]) for row in subset])
            yerr = np.array([
                float(row["normalized_second_moment_ci95_halfwidth"]) for row in subset
            ])
            ax.errorbar(x, y, yerr=yerr, capsize=2.0, **backend_style[backend])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.invert_xaxis()
        ax.xaxis.set_major_locator(FixedLocator([0.20, 0.10, 0.05]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        ax.xaxis.set_minor_formatter(FuncFormatter(lambda *_: ""))
        ax.set_title(title)
        ax.set_xlabel(r"target ratio $\epsilon/\tau$")
        ax.tick_params(axis="both", which="both", labelsize=7.5)
    axes[0].set_ylabel("second moment / rate envelope")
    axes[-1].legend(loc="best")
    fig.subplots_adjust(wspace=0.34)
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "fig_rate_validation.pdf")
    fig.savefig(FIGURES / "fig_rate_validation.png", dpi=300)
    plt.close(fig)


def plot_allocation(rows: list[dict[str, str]]) -> None:
    style()
    allocations = ["matched", "uniform", "light-tail", "heavy-tail"]
    labels = ["matched", "uniform", r"$k=3$ law", r"$k=1.5$ law"]
    k_values = [3.0, 2.0, 1.5]
    x = np.arange(len(k_values))
    width = 0.19
    colors = ["#202020", "#777777", "#B0B0B0", "#D0D0D0"]
    hatches = [None, "//", "..", "xx"]
    fig, ax = plt.subplots(figsize=(3.25, 2.35))
    for index, (allocation, label) in enumerate(zip(allocations, labels)):
        values = []
        for k in k_values:
            row = next(
                row for row in rows
                if float(row["k"]) == k and row["allocation"] == allocation
            )
            values.append(float(row["ratio_to_matched"]))
        ax.bar(
            x + (index - 1.5) * width,
            values,
            width=width,
            color=colors[index],
            hatch=hatches[index],
            edgecolor="black",
            linewidth=0.45,
            label=label,
        )
    ax.axhline(1.0, color="black", linewidth=0.65)
    ax.set_yscale("log")
    ax.set_ylim(0.9, 50.0)
    ax.set_xticks(x)
    ax.set_xticklabels([r"$k=3$", r"$k=2$", r"$k=1.5$"])
    ax.set_ylabel("variance objective / matched")
    # Keep the legend away from the deliberately tall mismatched-allocation
    # bar at k=1.5; the upper-left quadrant contains no large bars.
    ax.legend(ncol=2, loc="upper left")
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "fig_allocation_ablation.pdf")
    fig.savefig(FIGURES / "fig_allocation_ablation.png", dpi=300)
    plt.close(fig)


def main() -> None:
    plot_scaling(read_csv(RESULTS / "scaling.csv"))
    plot_allocation(read_csv(RESULTS / "allocation.csv"))
    print(FIGURES / "fig_rate_validation.pdf")
    print(FIGURES / "fig_allocation_ablation.pdf")


if __name__ == "__main__":
    main()
