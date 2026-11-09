"""Log-scale axes with a visible lane for zero counts, and label placement for scatter plots."""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes

from getwork.taxonomy import viz_data

ZERO_LANE = 0.5  # where a count of 0 is drawn on a log axis, labelled "0"
LIMIT_PAD_LOW, LIMIT_PAD_HIGH = 1.5, 1.8
LABEL_CHAR_FRACTION = 0.0075  # axes fraction per character at font size 7
LABEL_HEIGHT_FRACTION = 0.03
LABEL_LIFT = 0.025


def log_position(values: np.ndarray) -> np.ndarray:
    """Plot positions for non-negative counts: zeros move to ZERO_LANE, the rest are kept."""
    array = np.asarray(values, dtype=np.float64)
    return np.where(array == 0, ZERO_LANE, array)


def _limits_and_ticks(positions: np.ndarray) -> tuple[float, float, list[float], list[str]]:
    low, high = float(positions.min()) / LIMIT_PAD_LOW, float(positions.max()) * LIMIT_PAD_HIGH
    decades = [m * 10.0**k for k in range(0, 9) for m in (1, 2, 5) if low <= m * 10.0**k <= high]
    ticks, labels = list(decades), [f"{t:g}" for t in decades]
    if positions.min() == ZERO_LANE:
        ticks, labels = [ZERO_LANE, *ticks], ["0", *labels]
    return low, high, ticks, labels


def set_log_axes(ax: Axes, *, x: np.ndarray, y: np.ndarray) -> None:
    """Log axes whose limits start at the data minimum (x, y are plot positions)."""
    for axis, positions in (("x", x), ("y", y)):
        low, high, ticks, labels = _limits_and_ticks(positions)
        getattr(ax, f"set_{axis}scale")("log")
        getattr(ax, f"set_{axis}lim")(low, high)
        getattr(ax, f"set_{axis}ticks")(ticks, labels=labels)
        ax.minorticks_off()


def _axes_fraction(ax: Axes, points: np.ndarray) -> np.ndarray:
    (x_low, x_high), (y_low, y_high) = ax.get_xlim(), ax.get_ylim()
    return np.stack(
        [
            (np.log10(points[:, 0]) - np.log10(x_low)) / (np.log10(x_high) - np.log10(x_low)),
            (np.log10(points[:, 1]) - np.log10(y_low)) / (np.log10(y_high) - np.log10(y_low)),
        ],
        axis=1,
    )


def annotate_spread(ax: Axes, *, points: np.ndarray, texts: list[str], fontsize: int) -> None:
    """Label points on a log-log axes, moving labels apart with leader lines (limits set first)."""
    if not texts:
        return
    anchors = _axes_fraction(ax, points)
    widths = np.array([len(t) * LABEL_CHAR_FRACTION for t in texts])
    placed = viz_data.spread_labels(
        anchors=anchors + [0.0, LABEL_LIFT], widths=widths, height=LABEL_HEIGHT_FRACTION
    )
    for text, point, target in zip(texts, points, placed):
        ax.annotate(
            text, xy=tuple(point), xytext=tuple(np.clip(target, 0.03, 0.97)),
            textcoords="axes fraction", fontsize=fontsize, ha="center", va="center",
            arrowprops={"arrowstyle": "-", "color": "#888888", "lw": 0.4},
        )
