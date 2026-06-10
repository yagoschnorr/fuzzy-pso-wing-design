"""Avaliador Mamdani vetorizado em NumPy (apenas para o laço interno da calibração).

Responsabilidade
----------------
Reproduzir a inferência Mamdani (mín/máx/centróide) do `scikit-fuzzy` de forma VETORIZADA sobre um
lote de entradas (N, 3), evitando o custo de reconstruir um `ControlSystem` e simular ponto a ponto a
cada avaliação da função objetivo do PSO (ver seção "Desempenho" do plano do Portão B).

O `scikit-fuzzy` permanece o sistema OFICIAL (relatório/conferência via `dfpo.fuzzy.system`); este
módulo é uma otimização numérica interna, validada contra ele em `tests/test_pso.py`.

Semântica reproduzida (defaults do scikit-fuzzy):
    - ativação do antecedente : interpolação da MF no universo (`np.interp` == `fuzz.interp_membership`)
    - conjunção (E)           : mínimo
    - implicação Mamdani      : recorte (mín) do consequente pela força da regra
    - agregação entre regras   : máximo
    - defuzzificação          : centróide
"""
from __future__ import annotations

import numpy as np

from ..config import MF_PARAMS, S_RESOLUTION, UNIVERSES
from .membership import make_mf
from .rules import RULE_TABLE, RuleSpec
from .variables import INPUT_STEP


class FastMamdani:
    """Avaliador Mamdani vetorizado para um conjunto fixo de regras.

    Os universos são fixos (definidos pelos limites de `config.UNIVERSES`); apenas as curvas das MFs
    mudam quando `mf_params` muda — por isso são recomputadas a cada instância (barato).
    """

    def __init__(self, mf_params: dict | None = None,
                 rule_table: tuple[RuleSpec, ...] = RULE_TABLE) -> None:
        params = mf_params or MF_PARAMS
        self.rule_table = rule_table

        # Universos (idênticos aos de fuzzy/variables.py).
        self._u = {
            name: np.arange(lo, hi + step, step)
            for name, step in (("x", INPUT_STEP), ("y", INPUT_STEP),
                               ("z", INPUT_STEP), ("S", S_RESOLUTION))
            for lo, hi in [UNIVERSES[name]]
        }
        self._S_u = self._u["S"]

        # Curvas das MFs (recomputadas para os params correntes).
        self._mf = {
            var: {term: make_mf(self._u[var], p) for term, p in params[var].items()}
            for var in ("x", "y", "z", "S")
        }

    def infer_batch(self, inputs: np.ndarray) -> np.ndarray:
        """Avalia S para um lote de entradas (N, 3) -> array (N,) em [0, 100]."""
        inputs = np.asarray(inputs, dtype=float)
        xs, ys, zs = inputs[:, 0], inputs[:, 1], inputs[:, 2]
        n = inputs.shape[0]

        # Pertinências de cada termo nas entradas (interpolação no universo).
        mu = {
            "x": {t: np.interp(xs, self._u["x"], c) for t, c in self._mf["x"].items()},
            "y": {t: np.interp(ys, self._u["y"], c) for t, c in self._mf["y"].items()},
            "z": {t: np.interp(zs, self._u["z"], c) for t, c in self._mf["z"].items()},
        }

        agg = np.zeros((n, self._S_u.size))
        for spec in self.rule_table:
            strength = np.minimum.reduce([mu["x"][spec.x], mu["y"][spec.y], mu["z"][spec.z]])
            clipped = np.minimum(strength[:, None], self._mf["S"][spec.s][None, :])
            agg = np.maximum(agg, clipped)

        num = (agg * self._S_u[None, :]).sum(axis=1)
        den = agg.sum(axis=1)
        # Quando nenhuma regra dispara (den==0), usa o centro do universo de S (fallback neutro).
        out = np.where(den > 0, num / np.where(den > 0, den, 1.0), self._S_u.mean())
        return out

    def infer(self, x: float, y: float, z: float) -> float:
        """Conveniência: avalia um único ponto."""
        return float(self.infer_batch(np.array([[x, y, z]]))[0])
