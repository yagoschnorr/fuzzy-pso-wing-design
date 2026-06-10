"""Busca operacional — feature de produto.

Responsabilidade
----------------
Dado um valor de carga útil `z`, encontrar a geometria `(x, y)` que MAXIMIZA o score fuzzy
`S(x, y, z)` usando o PSO (pso/swarm.minimize sobre o custo = -S).

Variáveis de decisão : (x, y)
Restrições           : x in [0,5], y in [0,4]  (limites de caixa)
Sentido              : maximizar S  ->  minimizar (-S)

ESTADO (Portão A): STUB.
"""
from __future__ import annotations

from typing import NamedTuple

from ..config import DEFAULT_PSO, PSOParams, UNIVERSES
from ..fuzzy.system import FuzzySystem
from . import swarm


class OperationResult(NamedTuple):
    x: float
    y: float
    score: float
    pso: "swarm.PSOResult"


def optimize_geometry(
    fs: FuzzySystem,
    z: float,
    params: PSOParams = DEFAULT_PSO,
    seed: int | None = None,
) -> OperationResult:
    """Acha (x, y) que maximiza S(x, y, z) minimizando o custo -S via PSO.

    Reutiliza um único `FuzzySystem` já `build()`-ado (MFs fixas) → cada avaliação é barata.
    """
    def custo(p):  # p = [x, y]
        return -fs.infer(float(p[0]), float(p[1]), z)

    lb = [UNIVERSES["x"][0], UNIVERSES["y"][0]]
    ub = [UNIVERSES["x"][1], UNIVERSES["y"][1]]
    res = swarm.minimize(custo, lb, ub, params, seed)
    x, y = res.best_position
    return OperationResult(x=float(x), y=float(y), score=-res.best_cost, pso=res)
