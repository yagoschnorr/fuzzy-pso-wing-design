"""Funções de pertinência (MFs) — wrappers finos sobre scikit-fuzzy.

Responsabilidade
----------------
Centralizar a criação das MFs triangulares/trapezoidais a partir dos parâmetros de `config.MF_PARAMS`,
delegando o cálculo numérico a `skfuzzy` (skfuzzy.trimf / skfuzzy.trapmf). Mantém um único ponto para
validar restrições de ordenação/limites usadas pela calibração (Portão B).

ESTADO (Portão A): STUB (assinaturas + pseudocódigo).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

import skfuzzy as fuzz


def make_mf(universe: np.ndarray, params: Sequence[float]) -> np.ndarray:
    """Cria a curva de pertinência sobre `universe`.

    Regra: len(params) == 3 -> triangular (trimf); len(params) == 4 -> trapezoidal (trapmf).
    """
    params = list(params)
    if len(params) == 3:
        return fuzz.trimf(universe, params)
    if len(params) == 4:
        return fuzz.trapmf(universe, params)
    raise ValueError(
        f"params deve ter 3 (trimf) ou 4 (trapmf) elementos; recebido {len(params)}: {params}"
    )


def is_valid_breakpoints(params: Sequence[float], lo: float, hi: float) -> bool:
    """Valida as restrições da calibração: ordenação não-decrescente e dentro de [lo, hi]."""
    return all(lo <= p <= hi for p in params) and all(
        a <= b for a, b in zip(params, params[1:])
    )
