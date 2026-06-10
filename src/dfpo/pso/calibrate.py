"""Calibração do sistema fuzzy por PSO — NÚCLEO CIENTÍFICO (Parte 2 + ponto extra).

Responsabilidade
----------------
Ajustar automaticamente os parâmetros do sistema fuzzy (breakpoints das funções de pertinência e/ou
pesos das regras) para MINIMIZAR o erro (MSE) entre a saída fuzzy e uma referência, comparando o
desempenho ANTES e DEPOIS da calibração.

Formulação do problema de otimização (exigida pelas laudas):
    Variáveis de decisão : breakpoints das MFs (real) [+ pesos das regras (real)]
                           [extensão: conclusões das regras (inteiro) — mixed-coding do artigo-base]
    Função objetivo      : MSE(S_fuzzy(θ), S_ref) sobre o conjunto de referência   -> MINIMIZAR
    Restrições           : ordenação dos breakpoints (não-decrescente) e limites dos universos
    Referência           : superfície analítica normalizada (x,y) + dataset sintético p/ z
                           (ver dfpo/reference/target.py)

Base científica
---------------
Kacimi, M.A. et al. (2020). "New mixed-coding PSO algorithm for a self-adaptive and automatic learning
of Mamdani fuzzy rules." Engineering Applications of Artificial Intelligence, 89, 103417.
  - codifica MFs + fatores de escala (real) e conclusões das regras (inteiro) na MESMA partícula;
  - "monitoring function" + limiar auto-adaptativo movem as conclusões inteiras de forma gradual;
  - objetivo = MSE.
Plano de adaptação (Portão B): começar pela versão REAL-CODED (MFs/pesos); a extensão mixed-coding
(conclusões inteiras + monitoring function) entra como incremento.

ESTADO (Portão A): STUB. Nenhuma codificação/decodificação ou função objetivo implementada.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..config import DEFAULT_PSO, MF_PARAMS, PSOParams, UNIVERSES
from ..fuzzy.fast_eval import FastMamdani
from ..fuzzy.membership import is_valid_breakpoints
from . import swarm  # noqa: F401  (tipo PSOResult referenciado em CalibrationResult)

# Ordem fixa e documentada das variáveis no vetor de decisão.
_VAR_ORDER = ("x", "y", "z", "S")


@dataclass
class CalibrationResult:
    mf_params_before: dict
    mf_params_after: dict
    mse_before: float
    mse_after: float
    improvement_pct: float
    pso: "swarm.PSOResult"


def _free_layout(template: dict) -> list[tuple[str, str, int]]:
    """Lista os breakpoints OTIMIZÁVEIS em ordem fixa.

    Um breakpoint é mantido FIXO quando está colado a uma borda do universo (== lo ou == hi);
    os demais (pontos interiores das MFs) são livres para a calibração. Isso reduz a
    dimensionalidade e preserva a forma das MFs nas extremidades (Curta/Longa, Baixa/Excelente).
    """
    layout: list[tuple[str, str, int]] = []
    for var in _VAR_ORDER:
        lo, hi = UNIVERSES[var]
        for term, params in template[var].items():
            for idx, val in enumerate(params):
                if val != lo and val != hi:
                    layout.append((var, term, idx))
    return layout


def encode(mf_params: dict) -> np.ndarray:
    """Achata os breakpoints interiores das MFs num vetor de decisão para o PSO."""
    layout = _free_layout(mf_params)
    return np.array([mf_params[var][term][idx] for (var, term, idx) in layout], dtype=float)


def decode(vector: np.ndarray, template: dict) -> dict:
    """Reconstrói o dicionário de MFs (inverso de `encode`), reparando as restrições.

    Reparo: por termo, clipa todos os breakpoints aos limites do universo e os ordena de forma
    não-decrescente (garante MFs válidas para qualquer vetor proposto pelo PSO).
    """
    layout = _free_layout(template)
    mf = copy.deepcopy(template)
    for k, (var, term, idx) in enumerate(layout):
        mf[var][term][idx] = float(vector[k])

    for var in _VAR_ORDER:
        lo, hi = UNIVERSES[var]
        for term, params in mf[var].items():
            repaired = sorted(min(max(p, lo), hi) for p in params)
            assert is_valid_breakpoints(repaired, lo, hi)
            mf[var][term] = repaired
    return mf


def make_objective(reference_inputs: np.ndarray, reference_outputs: np.ndarray,
                   template: dict) -> Callable[[np.ndarray], float]:
    """Cria a função objetivo MSE(θ) usada pelo PSO (avaliador Mamdani vetorizado).

    Usa `FastMamdani` (mín/máx/centróide vetorizado) no laço interno por desempenho; o sistema
    oficial scikit-fuzzy é equivalente (validado em tests/test_pso.py).
    """
    reference_outputs = np.asarray(reference_outputs, dtype=float)

    def objective(vector: np.ndarray) -> float:
        mf = decode(vector, template)
        preds = FastMamdani(mf_params=mf).infer_batch(reference_inputs)
        return float(np.mean((preds - reference_outputs) ** 2))

    return objective


def calibrate(
    reference_inputs: np.ndarray,
    reference_outputs: np.ndarray,
    base_mf_params: dict | None = None,
    params: PSOParams = DEFAULT_PSO,
    seed: int | None = None,
) -> CalibrationResult:
    """Calibra as MFs via PSO e reporta MSE antes/depois.

    O vetor de decisão são os breakpoints interiores das MFs (ver `_free_layout`); os limites de
    caixa de cada componente são os limites do universo da respectiva variável. Garante-se que a
    configuração retornada nunca seja pior que o ponto de partida (mantém-se o melhor conhecido).
    """
    template = base_mf_params or MF_PARAMS
    layout = _free_layout(template)
    lb = np.array([UNIVERSES[var][0] for (var, _, _) in layout], dtype=float)
    ub = np.array([UNIVERSES[var][1] for (var, _, _) in layout], dtype=float)

    objective = make_objective(reference_inputs, reference_outputs, template)
    mse_before = objective(encode(template))

    res = swarm.minimize(objective, lb, ub, params, seed)

    if res.best_cost <= mse_before:
        mf_after = decode(res.best_position, template)
        mse_after = res.best_cost
    else:
        # PSO não superou o ponto de partida: mantém o template (melhor conhecido).
        mf_after = copy.deepcopy(template)
        mse_after = mse_before

    improvement_pct = 100.0 * (mse_before - mse_after) / mse_before if mse_before > 0 else 0.0
    return CalibrationResult(
        mf_params_before=copy.deepcopy(template),
        mf_params_after=mf_after,
        mse_before=mse_before,
        mse_after=mse_after,
        improvement_pct=improvement_pct,
        pso=res,
    )
