"""Curva de sensibilidade de >= 1 função de pertinência (evidência exigida pela Parte 1).

Responsabilidade
----------------
Variar um parâmetro de uma MF (ex.: o pico de x:Media) dentro de uma faixa e medir o efeito na saída S
para uma entrada de operação fixa, produzindo a curva de sensibilidade.

ESTADO (Portão A): STUB.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt

from .. import config
from ..fuzzy.system import FuzzySystem


def sensitivity_curve(
    variable: str,
    term: str,
    param_index: int,
    sweep_values: Sequence[float],
    operating_point: tuple[float, float, float],
    save_path: str | Path | None = None,
):
    """Mede S em função de um breakpoint da MF (variable[term][param_index]).

    Parameters
    ----------
    variable : 'x' | 'y' | 'z' | 'S'
    term     : nome do termo linguístico (ex.: 'Media')
    param_index : índice do breakpoint a variar
    sweep_values : valores a testar para o breakpoint
    operating_point : (x, y, z) fixo onde S é avaliado

    Retorna (sweep_values, S_out, fig).
    """
    sweep_values = list(sweep_values)
    S_out: list[float] = []
    for v in sweep_values:
        mf = copy.deepcopy(config.MF_PARAMS)
        mf[variable][term][param_index] = v
        fs = FuzzySystem(mf_params=mf).build()
        S_out.append(fs.infer(*operating_point))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(sweep_values, S_out, marker="o")
    ax.set_xlabel(f"{variable}['{term}'] breakpoint[{param_index}]")
    ax.set_ylabel("S [%]")
    ax.set_title(
        f"Sensibilidade de S a {variable}['{term}'][{param_index}] "
        f"em (x,y,z)={operating_point}"
    )
    ax.grid(True, alpha=0.3)
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return sweep_values, S_out, fig
