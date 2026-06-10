"""Variáveis linguísticas do sistema fuzzy (Antecedents/Consequent do scikit-fuzzy).

Responsabilidade
----------------
Construir as variáveis fuzzy a partir dos universos (`config.UNIVERSES`) e dos parâmetros das MFs
(`config.MF_PARAMS`):

    Entradas (Antecedent):
        x -- envergadura [0,5] m   -> {Curta, Media, Longa}
        y -- corda       [0,4] m   -> {Estreita, Media, Larga}
        z -- carga útil  [0,10] kg -> {Leve, Padrao, Pesada}
    Saída (Consequent):
        S -- score de eficiência [0,100] % -> {Baixa, Regular, Excelente}

ESTADO (Portão A): STUB.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
import skfuzzy.control as ctrl

from ..config import MF_PARAMS, S_RESOLUTION, UNIVERSES
from .membership import make_mf

# Passo fino dos universos das entradas (resolução das MFs de x, y, z).
INPUT_STEP = 0.01


class FuzzyVariables(NamedTuple):
    """Contêiner das 3 entradas + 1 saída (objetos ctrl.Antecedent / ctrl.Consequent)."""
    x: object  # ctrl.Antecedent
    y: object  # ctrl.Antecedent
    z: object  # ctrl.Antecedent
    S: object  # ctrl.Consequent


def build_variables(mf_params: dict | None = None) -> FuzzyVariables:
    """Cria as variáveis fuzzy e popula suas MFs.

    Parameters
    ----------
    mf_params : dict | None
        Parâmetros das MFs. Se None, usa `config.MF_PARAMS` (ponto de partida). Na calibração
        (Portão B), o PSO passará um dicionário com os breakpoints candidatos.

    """
    params = mf_params or MF_PARAMS

    def universe(name: str, step: float) -> np.ndarray:
        lo, hi = UNIVERSES[name]
        return np.arange(lo, hi + step, step)

    x = ctrl.Antecedent(universe("x", INPUT_STEP), "x")
    y = ctrl.Antecedent(universe("y", INPUT_STEP), "y")
    z = ctrl.Antecedent(universe("z", INPUT_STEP), "z")
    S = ctrl.Consequent(universe("S", S_RESOLUTION), "S")

    for var, name in ((x, "x"), (y, "y"), (z, "z"), (S, "S")):
        for label, mf_params_list in params[name].items():
            var[label] = make_mf(var.universe, mf_params_list)

    return FuzzyVariables(x, y, z, S)
