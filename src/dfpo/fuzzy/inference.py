"""Motor de inferência Mamdani (scikit-fuzzy).

Responsabilidade
----------------
Montar o ControlSystem do scikit-fuzzy a partir das variáveis + regras e configurar os operadores
exigidos pelas laudas:
    - conjunção (E)      : MÍNIMO
    - agregação          : MÁXIMO
    - defuzzificação     : CENTRÓIDE

ESTADO (Portão A): STUB.
"""
from __future__ import annotations

import skfuzzy.control as ctrl


def build_control_system(rules: list) -> object:
    """Cria o ctrl.ControlSystem com a lista de regras."""
    return ctrl.ControlSystem(rules)


def make_simulation(control_system: object) -> object:
    """Cria uma ControlSystemSimulation para avaliar entradas.

    Nota: scikit-fuzzy usa por padrão mín (conjunção), máx (agregação) e centróide (defuzz.),
    exatamente os operadores exigidos pelas laudas.
    """
    return ctrl.ControlSystemSimulation(control_system)


def infer_once(simulation: object, x: float, y: float, z: float) -> float:
    """Avalia o sistema para uma entrada (x, y, z) e retorna o score S defuzzificado."""
    simulation.input["x"] = x
    simulation.input["y"] = y
    simulation.input["z"] = z
    simulation.compute()
    return float(simulation.output["S"])
