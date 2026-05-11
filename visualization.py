"""Matplotlib visualizations for RF network simulation results."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "qthack-mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "qthack-cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

DB_FLOOR = 1e-15


def plot_s_matrix_heatmap(
    s_matrix: np.ndarray,
    output_path: str | Path | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot S-matrix magnitude in dB using a readable dark theme."""

    matrix = _validated_s_matrix(s_matrix)
    magnitude_db = 20.0 * np.log10(np.maximum(np.abs(matrix), DB_FLOOR))
    port_labels = [f"P{port}" for port in range(1, matrix.shape[0] + 1)]

    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(figsize=(7.0, 5.8), constrained_layout=True)
        fig.patch.set_facecolor("#111318")
        ax.set_facecolor("#151922")

        image = ax.imshow(magnitude_db, cmap="magma", aspect="equal")
        colorbar = fig.colorbar(image, ax=ax, shrink=0.86)
        colorbar.set_label("|Sij| (dB)", color="#f2f4f8")
        colorbar.ax.tick_params(colors="#d8dee9")

        ax.set_title("S-Matrix Magnitude", color="#f8fafc", pad=14, fontsize=15)
        ax.set_xlabel("Input Port", color="#d8dee9", labelpad=10)
        ax.set_ylabel("Output Port", color="#d8dee9", labelpad=10)
        ax.set_xticks(np.arange(matrix.shape[1]), labels=port_labels)
        ax.set_yticks(np.arange(matrix.shape[0]), labels=port_labels)
        ax.tick_params(colors="#d8dee9")

        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                value = magnitude_db[row, col]
                text_color = "#111318" if value > -8.0 else "#f8fafc"
                ax.text(
                    col,
                    row,
                    f"{value:.1f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=11,
                    fontweight="bold",
                )

        _save_if_requested(fig, output_path)
        return fig, ax


def plot_output_power_bar(
    output_power: Sequence[float] | np.ndarray,
    output_path: str | Path | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot normalized output power per port."""

    powers = np.asarray(output_power, dtype=float)
    if powers.ndim != 1:
        raise ValueError("output_power must be a 1D vector")

    port_labels = [f"P{port}" for port in range(1, powers.shape[0] + 1)]

    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)
        fig.patch.set_facecolor("#111318")
        ax.set_facecolor("#151922")

        bars = ax.bar(
            port_labels,
            powers,
            color=["#66e3ff", "#a6e22e", "#ffb86c"][: powers.shape[0]],
            edgecolor="#f8fafc",
            linewidth=0.9,
        )

        ax.set_title("Output Power Per Port", color="#f8fafc", pad=14, fontsize=15)
        ax.set_xlabel("Port", color="#d8dee9", labelpad=10)
        ax.set_ylabel("Normalized Power |b|^2", color="#d8dee9", labelpad=10)
        ax.tick_params(colors="#d8dee9")
        ax.grid(axis="y", color="#3a4252", alpha=0.55, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_ylim(0.0, max(float(np.max(powers)) * 1.18, 1e-3))

        for bar, power in zip(bars, powers, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"{power:.4f}",
                ha="center",
                va="bottom",
                color="#f8fafc",
                fontsize=10,
                fontweight="bold",
            )

        _save_if_requested(fig, output_path)
        return fig, ax


def animate_signal_flow(
    output_waves: Sequence[complex] | np.ndarray,
    output_path: str | Path | None = None,
    interval_ms: int = 120,
) -> FuncAnimation:
    """Create a simple three-port circulating signal-flow animation."""

    waves = np.asarray(output_waves, dtype=np.complex128)
    if waves.shape != (3,):
        raise ValueError("signal-flow animation requires exactly three output waves")

    angles = np.deg2rad([90.0, 210.0, 330.0])
    nodes = np.column_stack((np.cos(angles), np.sin(angles)))
    path_order = (0, 1, 2, 0)
    path_strength = np.abs(waves) / max(float(np.max(np.abs(waves))), DB_FLOOR)

    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(figsize=(5.8, 5.8), constrained_layout=True)
        fig.patch.set_facecolor("#111318")
        ax.set_facecolor("#151922")
        ax.set_title("Signal Flow", color="#f8fafc", pad=12, fontsize=15)
        ax.set_xlim(-1.35, 1.35)
        ax.set_ylim(-1.25, 1.35)
        ax.set_aspect("equal")
        ax.axis("off")

        for idx, (x_pos, y_pos) in enumerate(nodes, start=1):
            ax.scatter(x_pos, y_pos, s=850, color="#222936", edgecolor="#66e3ff")
            ax.text(x_pos, y_pos, f"P{idx}", ha="center", va="center", fontsize=13)

        for start, end in zip(path_order[:-1], path_order[1:], strict=True):
            ax.plot(
                [nodes[start, 0], nodes[end, 0]],
                [nodes[start, 1], nodes[end, 1]],
                color="#3a4252",
                linewidth=2.0,
            )

        marker = ax.scatter([], [], s=260, color="#ffb86c", edgecolor="#f8fafc")

        def update(frame: int) -> tuple:
            segment = frame % 3
            start = path_order[segment]
            end = path_order[segment + 1]
            fraction = (frame % 30) / 29.0
            position = (1.0 - fraction) * nodes[start] + fraction * nodes[end]
            marker.set_offsets([position])
            marker.set_sizes([180.0 + 220.0 * path_strength[end]])
            return (marker,)

        animation = FuncAnimation(fig, update, frames=90, interval=interval_ms, blit=True)
        if output_path is not None:
            _ensure_parent_dir(output_path)
            animation.save(output_path, writer="pillow", fps=max(1, 1000 // interval_ms))
        return animation


def _validated_s_matrix(s_matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(s_matrix, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("s_matrix must be a square 2D array")
    return matrix


def _save_if_requested(fig: plt.Figure, output_path: str | Path | None) -> None:
    if output_path is None:
        return
    _ensure_parent_dir(output_path)
    fig.savefig(output_path, dpi=180, facecolor=fig.get_facecolor())


def _ensure_parent_dir(output_path: str | Path) -> None:
    Path(output_path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
