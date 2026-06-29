"""Shared fragment-length plot annotations (150 bp short/long split)."""
from __future__ import annotations

SPLIT_BP = 150


def annotate_short_long_split(ax, plot_min: int, plot_max: int) -> None:
    """Vertical line at 150 bp with green arrows and short/long labels near y=0."""
    if not (plot_min < SPLIT_BP < plot_max):
        return

    ax.axvline(SPLIT_BP, color="black", linestyle="--", linewidth=1.2, zorder=5)

    ticks = [float(t) for t in ax.get_xticks() if plot_min <= t <= plot_max]
    if SPLIT_BP not in ticks:
        ticks.append(float(SPLIT_BP))
    ax.set_xticks(sorted(ticks))
    ax.set_xticklabels([f"{int(t)}" for t in ax.get_xticks()])

    ylo, yhi = ax.get_ylim()
    span = yhi - ylo if yhi > ylo else 1.0
    y_arrow = ylo + 0.03 * span
    y_text = ylo + 0.10 * span
    arrow_kw = dict(arrowstyle="->", color="#2E7D32", lw=2.8, mutation_scale=14)

    ax.annotate(
        "",
        xy=(130, y_arrow),
        xytext=(145, y_arrow),
        arrowprops=arrow_kw,
        zorder=6,
    )
    ax.annotate(
        "",
        xy=(170, y_arrow),
        xytext=(155, y_arrow),
        arrowprops=arrow_kw,
        zorder=6,
    )
    ax.text(
        140,
        y_text,
        "short fragments",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )
    ax.text(
        160,
        y_text,
        "long fragments",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )
