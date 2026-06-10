"""Configuração central do dfpo.

Aqui ficam **dados de configuração** (universos de discurso, parâmetros iniciais das funções de
pertinência, hiperparâmetros do PSO, sementes e caminhos). NÃO há lógica de algoritmo neste módulo —
ele é seguro de usar tanto pelos stubs quanto pelos mocks da API/dashboard.

Os parâmetros das MFs aqui são o PONTO DE PARTIDA (ver docs/funcoes_pertinencia.md). No Portão B o PSO
poderá calibrá-los; a versão calibrada será salva separadamente para a comparação antes/depois.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Caminhos
# --------------------------------------------------------------------------- #
ROOT_DIR = Path(__file__).resolve().parents[2]
EXPERIMENTS_DIR = ROOT_DIR / "experiments"
RESULTS_DIR = EXPERIMENTS_DIR / "results"
CONFIGS_DIR = EXPERIMENTS_DIR / "configs"
FIGURES_DIR = ROOT_DIR / "figures"

# --------------------------------------------------------------------------- #
# Universos de discurso (limites físicos das variáveis)
# --------------------------------------------------------------------------- #
UNIVERSES: dict[str, tuple[float, float]] = {
    "x": (0.0, 5.0),    # envergadura [m]
    "y": (0.0, 4.0),    # corda [m]
    "z": (0.0, 10.0),   # carga útil [kg]
    "S": (0.0, 100.0),  # score de eficiência [%]
}

UNITS: dict[str, str] = {"x": "m", "y": "m", "z": "kg", "S": "%"}

# --------------------------------------------------------------------------- #
# Parâmetros iniciais das funções de pertinência
#   forma "trimf"  -> [a, b, c]
#   forma "trapmf" -> [a, b, c, d]
# (ver docs/funcoes_pertinencia.md)
# --------------------------------------------------------------------------- #
MF_PARAMS: dict[str, dict[str, list[float]]] = {
    "x": {
        "Curta": [0.0, 0.0, 1.0, 2.0],
        "Media": [1.5, 2.5, 3.5],
        "Longa": [3.0, 4.0, 5.0, 5.0],
    },
    "y": {
        "Estreita": [0.0, 0.0, 0.8, 1.6],
        "Media": [1.2, 2.0, 2.8],
        "Larga": [2.4, 3.2, 4.0, 4.0],
    },
    "z": {
        "Leve": [0.0, 0.0, 2.0, 4.0],
        "Padrao": [3.0, 5.0, 7.0],
        "Pesada": [6.0, 8.0, 10.0, 10.0],
    },
    "S": {
        "Baixa": [0.0, 0.0, 20.0, 40.0],
        "Regular": [30.0, 50.0, 70.0],
        "Excelente": [60.0, 80.0, 100.0, 100.0],
    },
}

# Resolução do universo da saída para a defuzzificação por centróide (scikit-fuzzy).
S_RESOLUTION = 0.5

# --------------------------------------------------------------------------- #
# Baseline analítico: f(x, y) = 12x + 18y - 2x^2 - 3y^2 - x y
#   (coeficientes expostos como dados; a avaliação fica em baseline/analytic.py)
# --------------------------------------------------------------------------- #
BASELINE_COEFFS = {"x": 12.0, "y": 18.0, "x2": -2.0, "y2": -3.0, "xy": -1.0}
# Ótimo esperado (referência da especificação) — usado em testes (NÃO substitui o cálculo real):
BASELINE_OPTIMUM = {"x": 54.0 / 23.0, "y": 60.0 / 23.0}  # ~ (2.348, 2.609)


# --------------------------------------------------------------------------- #
# Hiperparâmetros do PSO (ajustáveis)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PSOParams:
    """Hiperparâmetros padrão do PSO (constrição de Clerc & Kennedy)."""
    n_particles: int = 30
    n_iterations: int = 50
    w: float = 0.729          # inércia
    c1: float = 1.494         # coeficiente cognitivo
    c2: float = 1.494         # coeficiente social
    v_max_frac: float = 0.2   # fração do range usada como limite de velocidade


DEFAULT_PSO = PSOParams()

# Sementes para os >= 5 experimentos independentes (rigor experimental das laudas).
DEFAULT_SEEDS: tuple[int, ...] = (1, 7, 13, 42, 99)

# Carga útil de referência [kg] usada na busca operacional e na superfície de controle.
OPERATIONAL_Z: float = 5.0


# --------------------------------------------------------------------------- #
# Cenários de teste (>= 6: baixo, médio, alto, fronteiriço, conflitante, crítico)
# Valores de entrada (x, y, z); a saída esperada será preenchida com execução real (Portão B).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Scenario:
    nome: str
    tipo: str
    x: float
    y: float
    z: float


TEST_SCENARIOS: tuple[Scenario, ...] = (
    Scenario("baixo",        "baixo",        0.5, 0.4, 1.0),
    Scenario("medio",        "medio",        2.4, 2.0, 5.0),
    Scenario("alto",         "alto",         3.8, 3.2, 9.0),
    Scenario("fronteirico",  "fronteirico",  2.5, 1.6, 5.0),
    Scenario("conflitante",  "conflitante",  0.8, 3.6, 5.0),
    Scenario("critico",      "critico",      0.5, 0.4, 9.5),
)
