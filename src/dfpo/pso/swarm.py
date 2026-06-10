"""PSO genérico (implementado manualmente).

Responsabilidade
----------------
Loop padrão do Particle Swarm Optimization com fator de constrição (w, c1, c2), limites de caixa
[lb, ub], limite de velocidade, sementes para reprodutibilidade e registro do histórico do gbest
(para as curvas de convergência) e do número de avaliações da função objetivo (nfe).

É o motor compartilhado pela CALIBRAÇÃO (pso/calibrate.py) e pela BUSCA OPERACIONAL (pso/operate.py).

Referência do método: Kennedy & Eberhart (1995); constrição de Clerc & Kennedy (2002).
O artigo-base (Kacimi et al., 2020) estende o PSO com codificação mista (mixed-coding) + monitoring
function para as conclusões inteiras das regras — ver pso/calibrate.py.

ESTADO (Portão A): STUB (assinaturas + pseudocódigo). NENHUM loop implementado.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from ..config import DEFAULT_PSO, PSOParams
from ..experiments import metrics

# Função objetivo: recebe um vetor de decisão (np.ndarray) e retorna um custo escalar (a MINIMIZAR).
ObjectiveFn = Callable[[np.ndarray], float]


@dataclass
class PSOResult:
    """Resultado de uma execução do PSO."""
    best_position: np.ndarray
    best_cost: float
    history: list[float] = field(default_factory=list)   # melhor custo por iteração (convergência)
    n_evaluations: int = 0                                 # nº de avaliações da função objetivo (nfe)
    convergence_iter: int | None = None                    # iteração em que estabilizou
    elapsed_s: float | None = None                         # tempo de execução [s]


def minimize(
    objective: ObjectiveFn,
    lb: Sequence[float],
    ub: Sequence[float],
    params: PSOParams = DEFAULT_PSO,
    seed: int | None = None,
) -> PSOResult:
    """Minimiza `objective` no hipercubo [lb, ub] via PSO (constrição de Clerc & Kennedy).

    Registra o histórico do gbest por iteração (curva de convergência), o nº de avaliações da
    função objetivo (nfe), a iteração de convergência e o tempo decorrido.
    """
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)
    dim = lb.size
    n = params.n_particles
    v_max = params.v_max_frac * (ub - lb)

    # Inicialização das partículas.
    X = rng.uniform(lb, ub, size=(n, dim))
    V = rng.uniform(-v_max, v_max, size=(n, dim))

    cost = np.array([objective(x) for x in X])
    nfe = n
    pbest = X.copy()
    pbest_cost = cost.copy()
    g = int(np.argmin(pbest_cost))
    gbest = pbest[g].copy()
    gbest_cost = float(pbest_cost[g])

    history = [gbest_cost]

    for _ in range(params.n_iterations):
        r1 = rng.random(size=(n, dim))
        r2 = rng.random(size=(n, dim))
        V = (params.w * V
             + params.c1 * r1 * (pbest - X)
             + params.c2 * r2 * (gbest - X))
        V = np.clip(V, -v_max, v_max)
        X = np.clip(X + V, lb, ub)

        cost = np.array([objective(x) for x in X])
        nfe += n

        # Atualiza melhores pessoais.
        improved = cost < pbest_cost
        pbest[improved] = X[improved]
        pbest_cost[improved] = cost[improved]

        # Atualiza melhor global.
        g = int(np.argmin(pbest_cost))
        if pbest_cost[g] < gbest_cost:
            gbest = pbest[g].copy()
            gbest_cost = float(pbest_cost[g])

        history.append(gbest_cost)

    convergence_iter = metrics.detect_convergence(history)
    elapsed_s = time.perf_counter() - t0

    return PSOResult(
        best_position=gbest,
        best_cost=gbest_cost,
        history=history,
        n_evaluations=nfe,
        convergence_iter=convergence_iter,
        elapsed_s=elapsed_s,
    )
