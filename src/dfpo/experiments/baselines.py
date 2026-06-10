"""Baselines de comparação para o PSO (exigidos pelas laudas).

Responsabilidade
----------------
Fornecer métodos de referência simples para situar o desempenho do PSO:
    - busca aleatória (random search): amostra N pontos uniformes e retorna o melhor;
    - busca gulosa (greedy/hill-climbing): a partir de um ponto, move-se p/ vizinhos melhores.

Mesma interface da função objetivo do PSO (custo a MINIMIZAR), para comparação justa (mesmo nfe).

ESTADO (Portão A): STUB.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

ObjectiveFn = Callable[[np.ndarray], float]


def random_search(
    objective: ObjectiveFn,
    lb: Sequence[float],
    ub: Sequence[float],
    n_evaluations: int,
    seed: int | None = None,
) -> tuple[np.ndarray, float, list[float]]:
    """Busca aleatória uniforme em [lb, ub].

    Returns: (melhor_posição, melhor_custo, histórico_do_melhor).
    O histórico tem comprimento `n_evaluations` (melhor custo acumulado a cada avaliação).
    """
    rng = np.random.default_rng(seed)
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)

    best_pos = None
    best_cost = np.inf
    history: list[float] = []
    for _ in range(int(n_evaluations)):
        p = rng.uniform(lb, ub)
        c = objective(p)
        if c < best_cost:
            best_cost = c
            best_pos = p
        history.append(best_cost)
    return best_pos, float(best_cost), history


def greedy_search(
    objective: ObjectiveFn,
    lb: Sequence[float],
    ub: Sequence[float],
    n_evaluations: int,
    step_frac: float = 0.05,
    seed: int | None = None,
) -> tuple[np.ndarray, float, list[float]]:
    """Hill-climbing simples com passo proporcional ao range.

    Parte de um ponto inicial aleatório; a cada passo propõe um vizinho (perturbação gaussiana com
    desvio `step_frac*(ub-lb)`), aceita se melhora; respeita o orçamento `n_evaluations`.
    Returns: (melhor_posição, melhor_custo, histórico_do_melhor).
    """
    rng = np.random.default_rng(seed)
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)
    step = step_frac * (ub - lb)

    current = rng.uniform(lb, ub)
    current_cost = objective(current)
    best_pos = current.copy()
    best_cost = current_cost
    history = [best_cost]

    for _ in range(int(n_evaluations) - 1):
        candidate = np.clip(current + rng.normal(0.0, step), lb, ub)
        c = objective(candidate)
        if c < current_cost:
            current = candidate
            current_cost = c
            if c < best_cost:
                best_cost = c
                best_pos = candidate.copy()
        history.append(best_cost)
    return best_pos, float(best_cost), history
