"""Superfície de controle 3D do sistema fuzzy (evidência exigida pela Parte 1).

Responsabilidade
----------------
Plotar S(x, y) para um z fixo (superfície 3D) e, opcionalmente, sobrepor/comparar com a superfície
analítica normalizada de referência.

ESTADO (Portão A): STUB.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registra a projeção '3d')

from ..fuzzy.system import FuzzySystem


def plot_control_surface(fs: FuzzySystem, z: float, n: int = 50,
                         save_path: str | Path | None = None):
    """Gera a superfície 3D S(x, y) | z."""
    X, Y, S = fs.infer_grid(z, n)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(projection="3d")
    surf = ax.plot_surface(X, Y, S, cmap="viridis", edgecolor="none")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("S [%]")
    ax.set_title(f"Superfície de controle fuzzy  S(x, y) | z = {z:g} kg")
    fig.colorbar(surf, ax=ax, shrink=0.6, label="S [%]")
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, ax
