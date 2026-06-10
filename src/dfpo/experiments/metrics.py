"""Métricas do rigor experimental (exigidas pelas duas laudas).

Responsabilidade
----------------
Agregar resultados de múltiplas execuções independentes e calcular:
    - qualidade da solução : melhor, pior, média, desvio-padrão do valor objetivo;
    - convergência         : iteração de convergência (estabilização do gbest);
    - custo computacional  : tempo de execução e nº de avaliações da função objetivo (nfe).

Sem lógica de algoritmo de otimização — apenas estatística sobre resultados já obtidos.

ESTADO (Portão A): STUB (assinaturas + pseudocódigo).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


@dataclass
class RunMetrics:
    """Métricas de UMA execução (uma semente)."""
    seed: int
    best_cost: float
    convergence_iter: int | None
    n_evaluations: int
    elapsed_s: float


@dataclass
class AggregateMetrics:
    """Estatística agregada sobre >= 5 execuções independentes."""
    n_runs: int
    best: float
    worst: float
    mean: float
    std: float
    mean_convergence_iter: float | None = None
    mean_elapsed_s: float | None = None
    mean_evaluations: float | None = None
    per_run: list[RunMetrics] = field(default_factory=list)


def detect_convergence(history: Sequence[float], tol: float = 1e-6, patience: int = 5) -> int | None:
    """Detecta a iteração em que o melhor custo estabiliza.

    Retorna a primeira iteração a partir da qual a melhora absoluta do gbest fica abaixo de `tol`
    por `patience` iterações consecutivas; `None` se nunca estabilizar. `history[k]` é o gbest ao
    final da iteração `k` (`history[0]` = gbest inicial).
    """
    if len(history) <= patience:
        return None
    streak = 0
    for i in range(1, len(history)):
        if abs(history[i - 1] - history[i]) < tol:
            streak += 1
            if streak >= patience:
                return i - patience + 1  # 1ª iteração da sequência estável
        else:
            streak = 0
    return None


def aggregate(runs: Sequence[RunMetrics]) -> AggregateMetrics:
    """Calcula best/worst/mean/std e médias de convergência/tempo/nfe sobre as execuções."""
    runs = list(runs)
    if not runs:
        raise ValueError("aggregate requer pelo menos uma execução (runs vazio).")

    costs = np.array([r.best_cost for r in runs], dtype=float)
    conv = [r.convergence_iter for r in runs if r.convergence_iter is not None]

    return AggregateMetrics(
        n_runs=len(runs),
        best=float(costs.min()),
        worst=float(costs.max()),
        mean=float(costs.mean()),
        std=float(costs.std(ddof=0)),
        mean_convergence_iter=float(np.mean(conv)) if conv else None,
        mean_elapsed_s=float(np.mean([r.elapsed_s for r in runs])),
        mean_evaluations=float(np.mean([r.n_evaluations for r in runs])),
        per_run=runs,
    )
