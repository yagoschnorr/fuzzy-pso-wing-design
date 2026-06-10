"""Facade do sistema fuzzy — ponto de entrada de alto nível.

Responsabilidade
----------------
Encapsular o pipeline completo (variáveis -> regras -> ControlSystem -> Simulation) numa interface
simples reutilizada pela API, pelo dashboard, pelo PSO (busca operacional/calibração) e pelas
visualizações:

    fs = FuzzySystem()              # usa MFs padrão (config.MF_PARAMS)
    s  = fs.infer(x=2.4, y=2.0, z=5.0)
    grid = fs.infer_grid(z=5.0)     # superfície S(x, y) p/ z fixo

ESTADO (Portão A): STUB. A construção real é montada no Portão B.
"""
from __future__ import annotations

import numpy as np

from ..config import MF_PARAMS
from . import inference, rules, variables


class FuzzySystem:
    """Sistema fuzzy Mamdani de 3 entradas (x, y, z) e 1 saída (S)."""

    def __init__(self, mf_params: dict | None = None,
                 rule_table: tuple | None = None) -> None:
        """Guarda a configuração; a montagem do ControlSystem ocorre em `build()` (Portão B)."""
        self.mf_params = mf_params or MF_PARAMS
        self.rule_table = rule_table or rules.RULE_TABLE
        self._cs = None   # ControlSystem (preenchido em build())
        self._vars = None  # FuzzyVariables (universos usados em infer_grid)
        self._sim = None  # ControlSystemSimulation (preenchido em build())

    def build(self) -> "FuzzySystem":
        """Monta variáveis, regras, ControlSystem e Simulation."""
        self._vars = variables.build_variables(self.mf_params)
        r = rules.build_rules(self._vars, self.rule_table)
        self._cs = inference.build_control_system(r)
        self._sim = inference.make_simulation(self._cs)
        return self

    def infer(self, x: float, y: float, z: float) -> float:
        """Retorna o score S [0,100] para a entrada (x, y, z).

        O ControlSystemSimulation do scikit-fuzzy acumula estado entre chamadas; para robustez
        criamos uma simulação nova a cada inferência a partir do ControlSystem cacheado.
        """
        if self._cs is None:
            self.build()
        sim = inference.make_simulation(self._cs)
        return inference.infer_once(sim, x, y, z)

    def infer_grid(self, z: float, n: int = 50) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Avalia S sobre uma grade (x, y) com z fixo — base p/ a superfície de controle 3D."""
        if self._cs is None:
            self.build()
        lo_x, hi_x = self._vars.x.universe.min(), self._vars.x.universe.max()
        lo_y, hi_y = self._vars.y.universe.min(), self._vars.y.universe.max()
        xs = np.linspace(lo_x, hi_x, n)
        ys = np.linspace(lo_y, hi_y, n)
        X, Y = np.meshgrid(xs, ys)
        S = np.empty_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                S[i, j] = self.infer(float(X[i, j]), float(Y[i, j]), z)
        return X, Y, S
