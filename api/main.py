"""FastAPI — Drone Fuzzy-PSO Optimizer (camada de API, OPCIONAL).

Expõe o núcleo `dfpo` via 4 endpoints:
    GET  /baseline        -> referência analítica
    POST /fuzzy/infer     -> score fuzzy S(x, y, z)
    POST /pso/optimize    -> busca operacional (dado z, maximizar S)
    POST /pso/calibrate   -> calibração das MFs por PSO (antes/depois)

ESTADO (Portão B): todos os endpoints usam o núcleo REAL (`placeholder=False`).

Subir:  uvicorn api.main:app --reload --port 8000
Docs:   http://localhost:8000/docs
"""
from __future__ import annotations

import sys
from pathlib import Path

# Torna `dfpo` (src/) importável sem instalar o pacote.
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from fastapi import FastAPI  # noqa: E402

from dfpo.baseline import analytic  # noqa: E402
from dfpo.config import PSOParams  # noqa: E402
from dfpo.fuzzy.system import FuzzySystem  # noqa: E402
from dfpo.pso import operate  # noqa: E402

from .schemas import (  # noqa: E402
    BaselineResponse,
    FuzzyInferRequest,
    FuzzyInferResponse,
    PSOCalibrateRequest,
    PSOCalibrateResponse,
    PSOOptimizeRequest,
    PSOOptimizeResponse,
)

app = FastAPI(
    title="Drone Fuzzy-PSO Optimizer API",
    description="Back-end do sistema fuzzy-evolutivo de otimização de asas para drones.",
    version="1.0.0",
)

# Sistema fuzzy construído uma única vez (MFs padrão) e reutilizado pelos endpoints.
_FS: FuzzySystem | None = None


def get_fuzzy_system() -> FuzzySystem:
    global _FS
    if _FS is None:
        _FS = FuzzySystem().build()
    return _FS


@app.get("/", tags=["meta"])
def root() -> dict:
    return {
        "service": "Drone Fuzzy-PSO Optimizer",
        "status": "operacional (Portão B) — respostas vêm do núcleo real",
        "endpoints": ["/baseline", "/fuzzy/infer", "/pso/optimize", "/pso/calibrate", "/docs"],
    }


@app.get("/baseline", response_model=BaselineResponse, tags=["baseline"])
def baseline() -> BaselineResponse:
    """Referência analítica f(x, y) — ponto crítico e classificação (valores reais)."""
    cp = analytic.critical_point()
    return BaselineResponse(
        formula="f(x,y) = 12x + 18y - 2x^2 - 3y^2 - x*y",
        optimum_x=cp.x,
        optimum_y=cp.y,
        f_optimum=cp.f_value,
        det_hessian=cp.det_hessian,
        f_xx=cp.f_xx,
        classification=cp.classification,
        placeholder=False,
    )


@app.post("/fuzzy/infer", response_model=FuzzyInferResponse, tags=["fuzzy"])
def fuzzy_infer(req: FuzzyInferRequest) -> FuzzyInferResponse:
    """Inferência fuzzy Mamdani S(x, y, z) usando o núcleo real."""
    score = get_fuzzy_system().infer(req.x, req.y, req.z)
    return FuzzyInferResponse(score=score, inputs=req, placeholder=False)


@app.post("/pso/optimize", response_model=PSOOptimizeResponse, tags=["pso"])
def pso_optimize(req: PSOOptimizeRequest) -> PSOOptimizeResponse:
    """Busca operacional via PSO: dado z, encontra (x, y) que maximiza S."""
    params = PSOParams(n_particles=req.n_particles, n_iterations=req.n_iterations)
    res = operate.optimize_geometry(get_fuzzy_system(), req.z, params=params, seed=req.seed)
    # history é o custo minimizado (-S); converte para S para exibição da convergência.
    convergence = [-c for c in res.pso.history]
    return PSOOptimizeResponse(
        best_x=res.x,
        best_y=res.y,
        best_score=res.score,
        convergence=convergence,
        placeholder=False,
    )


@app.post("/pso/calibrate", response_model=PSOCalibrateResponse, tags=["pso"])
def pso_calibrate(req: PSOCalibrateRequest) -> PSOCalibrateResponse:
    """Calibração das MFs por PSO (MSE antes/depois) via experimento de múltiplas sementes."""
    from dfpo.experiments import runner  # import tardio (pesado)

    params = PSOParams(n_particles=req.n_particles, n_iterations=req.n_iterations)
    report = runner.run_calibration_experiment(seeds=req.seeds, params=params)
    # Curva de convergência da primeira semente como representativa.
    first_label = next(iter(report.extra["histories"]))
    convergence = report.extra["histories"][first_label]
    return PSOCalibrateResponse(
        mse_before=report.extra["mse_before"],
        mse_after=report.extra["mse_after_mean"],
        improvement_pct=report.extra["improvement_pct_mean"],
        convergence=convergence,
        placeholder=False,
    )
