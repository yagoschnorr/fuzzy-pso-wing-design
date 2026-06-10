"""Curvas de convergência do PSO (evidência exigida pela Parte 2).

Responsabilidade
----------------
Plotar a evolução do melhor custo (gbest) ao longo das iterações, para uma ou várias execuções
(sementes), permitindo comparar PSO vs baselines (busca aleatória/gulosa).

ESTADO (Portão A): STUB.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt


def plot_convergence(
    histories: Mapping[str, Sequence[float]],
    save_path: str | Path | None = None,
    title: str = "Convergência do PSO",
):
    """Plota uma ou mais curvas de convergência.

    Parameters
    ----------
    histories : mapeamento rótulo -> histórico do melhor custo por iteração
                (ex.: {'PSO seed=42': [...], 'Random search': [...]})

    Retorna (fig, ax).
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, hist in histories.items():
        # baselines (busca aleatória/gulosa) em tracejado para distinguir do PSO.
        style = "--" if any(k in label.lower() for k in ("alea", "random", "gulos", "greedy")) else "-"
        ax.plot(range(len(hist)), hist, style, label=label, alpha=0.85)
    ax.set_xlabel("Iteração")
    ax.set_ylabel("Melhor custo (gbest)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, ax
