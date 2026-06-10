"""Modelos Pydantic (entradas/saídas) dos endpoints FastAPI.

Define o CONTRATO da API ponta a ponta. O campo `placeholder` indica se o valor veio de execução real
do núcleo (`False`, padrão no Portão B) ou de um mock (`True`).
"""
from __future__ import annotations

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# /baseline
# --------------------------------------------------------------------------- #
class BaselineResponse(BaseModel):
    formula: str = Field(..., description="Expressão de f(x, y).")
    optimum_x: float
    optimum_y: float
    f_optimum: float | None = None
    det_hessian: float
    f_xx: float
    classification: str
    placeholder: bool = False


# --------------------------------------------------------------------------- #
# /fuzzy/infer
# --------------------------------------------------------------------------- #
class FuzzyInferRequest(BaseModel):
    x: float = Field(..., ge=0, le=5, description="Envergadura [m]")
    y: float = Field(..., ge=0, le=4, description="Corda [m]")
    z: float = Field(..., ge=0, le=10, description="Carga útil [kg]")


class FuzzyInferResponse(BaseModel):
    score: float = Field(..., description="Score de eficiência S [0,100] %")
    inputs: FuzzyInferRequest
    placeholder: bool = False


# --------------------------------------------------------------------------- #
# /pso/optimize  (busca operacional: dado z, maximizar S)
# --------------------------------------------------------------------------- #
class PSOOptimizeRequest(BaseModel):
    z: float = Field(..., ge=0, le=10, description="Carga útil fixa [kg]")
    n_particles: int = 30
    n_iterations: int = 50
    seed: int | None = None


class PSOOptimizeResponse(BaseModel):
    best_x: float
    best_y: float
    best_score: float
    convergence: list[float] = Field(default_factory=list, description="gbest por iteração")
    placeholder: bool = False


# --------------------------------------------------------------------------- #
# /pso/calibrate  (calibração das MFs por PSO; antes/depois)
# --------------------------------------------------------------------------- #
class PSOCalibrateRequest(BaseModel):
    z_fixed: float = 5.0
    n_particles: int = 30
    n_iterations: int = 50
    seeds: list[int] = Field(default_factory=lambda: [1, 7, 13, 42, 99])


class PSOCalibrateResponse(BaseModel):
    mse_before: float
    mse_after: float
    improvement_pct: float
    convergence: list[float] = Field(default_factory=list)
    placeholder: bool = False
