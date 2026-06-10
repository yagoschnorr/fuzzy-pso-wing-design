"""Orquestrador dos experimentos (>= 5 sementes independentes).

Responsabilidade
----------------
Executar os experimentos do projeto de forma reprodutível e agregar as métricas:
    E1  Calibração do fuzzy por PSO (MSE antes/depois)   -> pso.calibrate
    E2  Busca operacional (dado z, maximizar S)          -> pso.operate
    +   Baselines (busca aleatória / gulosa)             -> experiments.baselines
    +   Agregação de métricas sobre as sementes          -> experiments.metrics

Salva resultados em `experiments/results/` (regra: todo número do relatório vem daqui).

ESTADO (Portão A): STUB.
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from ..config import (
    DEFAULT_PSO, DEFAULT_SEEDS, OPERATIONAL_Z, PSOParams, RESULTS_DIR, TEST_SCENARIOS, UNIVERSES,
)
from . import baselines, metrics


@dataclass
class ExperimentReport:
    name: str
    aggregate: "metrics.AggregateMetrics"
    extra: dict


def interpret_score(s: float) -> str:
    """Traduz o score S [0,100] para a etiqueta linguística dominante da saída."""
    if s < 40.0:
        return "Baixa"
    if s < 65.0:
        return "Regular"
    return "Excelente"


def run_scenarios(save_path: str | Path | None = None) -> list[dict]:
    """E3 — avalia os >= 6 cenários de teste com o FuzzySystem real e salva uma tabela CSV.

    Retorna a lista de linhas (dict) com entradas, S e interpretação. Se `save_path` for None,
    salva em `experiments/results/scenarios.csv`.
    """
    from ..fuzzy.system import FuzzySystem  # import tardio (evita custo se não usado)

    fs = FuzzySystem().build()
    rows: list[dict] = []
    for sc in TEST_SCENARIOS:
        s = fs.infer(sc.x, sc.y, sc.z)
        rows.append({
            "nome": sc.nome,
            "tipo": sc.tipo,
            "x": sc.x,
            "y": sc.y,
            "z": sc.z,
            "S": round(s, 2),
            "interpretacao": interpret_score(s),
        })

    out = Path(save_path) if save_path is not None else RESULTS_DIR / "scenarios.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def run_calibration_experiment(
    seeds: Sequence[int] = DEFAULT_SEEDS,
    params: PSOParams = DEFAULT_PSO,
) -> ExperimentReport:
    """E1 — roda a calibração para cada semente e agrega (inclui MSE antes/depois)."""
    from ..pso import calibrate
    from ..reference.target import build_reference

    ref_in, ref_out = build_reference()
    runs: list[metrics.RunMetrics] = []
    histories: dict[str, list[float]] = {}
    improvements: list[float] = []
    mse_before = None
    mse_after_per_seed: list[float] = []

    for seed in seeds:
        res = calibrate.calibrate(ref_in, ref_out, params=params, seed=seed)
        mse_before = res.mse_before  # determinístico (mesma referência/template)
        mse_after_per_seed.append(res.mse_after)
        improvements.append(res.improvement_pct)
        runs.append(metrics.RunMetrics(
            seed=seed,
            best_cost=res.mse_after,
            convergence_iter=res.pso.convergence_iter,
            n_evaluations=res.pso.n_evaluations,
            elapsed_s=res.pso.elapsed_s,
        ))
        histories[f"PSO seed={seed}"] = list(res.pso.history)

    agg = metrics.aggregate(runs)
    extra = {
        "n_reference_points": int(ref_in.shape[0]),
        "mse_before": mse_before,
        "mse_after_mean": float(sum(mse_after_per_seed) / len(mse_after_per_seed)),
        "improvement_pct_mean": float(sum(improvements) / len(improvements)),
        "histories": histories,
    }
    return ExperimentReport("calibracao", agg, extra)


def _subsample_to_iterations(history: list[float], n_particles: int, n_points: int) -> list[float]:
    """Reduz um histórico por-avaliação (baselines) a por-iteração, p/ alinhar com o eixo do PSO.

    Toma o melhor custo ao final de cada "geração" de `n_particles` avaliações.
    """
    out = []
    for g in range(n_points):
        idx = min((g + 1) * n_particles - 1, len(history) - 1)
        out.append(history[idx])
    return out


def run_operational_experiment(
    z: float = OPERATIONAL_Z,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    params: PSOParams = DEFAULT_PSO,
) -> ExperimentReport:
    """E2 — roda a busca operacional p/ um z, agrega e compara com baselines (mesmo nfe)."""
    from ..fuzzy.system import FuzzySystem
    from ..pso import operate

    fs = FuzzySystem().build()
    lb = [UNIVERSES["x"][0], UNIVERSES["y"][0]]
    ub = [UNIVERSES["x"][1], UNIVERSES["y"][1]]

    def cost(p):
        return -fs.infer(float(p[0]), float(p[1]), z)

    runs: list[metrics.RunMetrics] = []
    histories: dict[str, list[float]] = {}
    best_geometries: list[dict] = []
    random_best: list[float] = []
    greedy_best: list[float] = []
    n_points = params.n_iterations + 1

    for seed in seeds:
        r = operate.optimize_geometry(fs, z, params=params, seed=seed)
        nfe = r.pso.n_evaluations
        runs.append(metrics.RunMetrics(
            seed=seed,
            best_cost=-r.score,  # custo minimizado = -S
            convergence_iter=r.pso.convergence_iter,
            n_evaluations=nfe,
            elapsed_s=r.pso.elapsed_s,
        ))
        best_geometries.append({"seed": seed, "x": r.x, "y": r.y, "S": r.score})
        histories[f"PSO seed={seed}"] = list(r.pso.history)

        # Baselines com MESMO orçamento de avaliações (comparação justa).
        _, rc, rh = baselines.random_search(cost, lb, ub, nfe, seed=seed)
        _, gc, gh = baselines.greedy_search(cost, lb, ub, nfe, seed=seed)
        random_best.append(-rc)
        greedy_best.append(-gc)
        histories[f"Busca aleatoria seed={seed}"] = _subsample_to_iterations(rh, params.n_particles, n_points)
        histories[f"Busca gulosa seed={seed}"] = _subsample_to_iterations(gh, params.n_particles, n_points)

    agg = metrics.aggregate(runs)
    pso_best_S = [-rm.best_cost for rm in runs]
    extra = {
        "z": z,
        "best_geometries": best_geometries,
        "pso_best_S_mean": float(np.mean(pso_best_S)),
        "random_best_S_mean": float(np.mean(random_best)),
        "greedy_best_S_mean": float(np.mean(greedy_best)),
        "histories": histories,
    }
    return ExperimentReport("operacional", agg, extra)


def run_all(seeds: Sequence[int] = DEFAULT_SEEDS, params: PSOParams = DEFAULT_PSO) -> dict:
    """Executa todos os experimentos e devolve um dicionário de ExperimentReport."""
    return {
        "calibracao": run_calibration_experiment(seeds=seeds, params=params),
        "operacional": run_operational_experiment(z=OPERATIONAL_Z, seeds=seeds, params=params),
    }


# --------------------------------------------------------------------------- #
# Persistência (regra de coerência: todo número do relatório vem destes arquivos)
# --------------------------------------------------------------------------- #
def _json_default(o):
    """Serializa tipos NumPy em JSON."""
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Tipo não serializável: {type(o)}")


def save_report(report: ExperimentReport, out_dir: str | Path = RESULTS_DIR) -> None:
    """Persiste um ExperimentReport como JSON (métricas) + CSV (por semente e convergência)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    name = report.name
    extra = {k: v for k, v in report.extra.items() if k != "histories"}
    histories = report.extra.get("histories", {})

    # 1) Métricas agregadas + extra (sem históricos) em JSON.
    payload = {
        "name": name,
        "aggregate": {k: v for k, v in asdict(report.aggregate).items() if k != "per_run"},
        "extra": extra,
    }
    (out / f"{name}_metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8"
    )

    # 2) Métricas por semente em CSV.
    with (out / f"{name}_runs.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["seed", "best_cost", "convergence_iter", "n_evaluations", "elapsed_s"]
        )
        writer.writeheader()
        for rm in report.aggregate.per_run:
            writer.writerow(asdict(rm))

    # 3) Curvas de convergência (formato largo: iteração + uma coluna por série) em CSV.
    if histories:
        labels = list(histories.keys())
        n_rows = max(len(h) for h in histories.values())
        with (out / f"{name}_convergence.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["iteration", *labels])
            for i in range(n_rows):
                row = [i] + [
                    histories[l][i] if i < len(histories[l]) else "" for l in labels
                ]
                writer.writerow(row)


def save_all(reports: dict, out_dir: str | Path = RESULTS_DIR) -> None:
    """Persiste todos os relatórios de `run_all`."""
    for report in reports.values():
        save_report(report, out_dir)
