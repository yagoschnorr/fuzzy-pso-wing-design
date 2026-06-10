#!/usr/bin/env python3
"""Executa o protocolo experimental (>= 5 sementes) e salva métricas/curvas.

Interface (CLI) já definida; a execução real depende da lógica do Portão B.

Uso:
    python scripts/run_experiments.py --seeds 5 --config experiments/configs/default.yaml
    python scripts/run_experiments.py --experiment calibration --seeds 5
    python scripts/run_experiments.py --experiment operational --z 5.0 --seeds 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Torna `dfpo` (src/) importável sem instalar o pacote.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dfpo.config import DEFAULT_SEEDS, PSOParams, RESULTS_DIR  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Roda os experimentos do Drone Fuzzy-PSO Optimizer.")
    p.add_argument("--experiment", choices=["calibration", "operational", "all"], default="all",
                   help="Qual experimento rodar.")
    p.add_argument("--seeds", type=int, default=len(DEFAULT_SEEDS),
                   help="Número de execuções independentes (>= 5 exigido p/ método estocástico).")
    p.add_argument("--z", type=float, default=5.0, help="Carga útil fixa (busca operacional).")
    p.add_argument("--config", type=Path, default=ROOT / "experiments/configs/default.yaml",
                   help="Arquivo YAML de configuração (sementes, hiperparâmetros, cenários).")
    p.add_argument("--out", type=Path, default=RESULTS_DIR, help="Diretório de saída dos resultados.")
    return p.parse_args(argv)


def _load_config(path: Path) -> tuple[list[int], PSOParams]:
    """Carrega sementes e hiperparâmetros do PSO do YAML; cai nos defaults se ausente."""
    seeds = list(DEFAULT_SEEDS)
    params = PSOParams()
    if path.exists():
        import yaml
        cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        seeds = cfg.get("seeds", seeds)
        pso = cfg.get("pso", {})
        params = PSOParams(
            n_particles=pso.get("n_particles", params.n_particles),
            n_iterations=pso.get("n_iterations", params.n_iterations),
            w=pso.get("w", params.w),
            c1=pso.get("c1", params.c1),
            c2=pso.get("c2", params.c2),
            v_max_frac=pso.get("v_max_frac", params.v_max_frac),
        )
    return seeds, params


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    from dfpo.experiments import runner

    all_seeds, params = _load_config(args.config)
    seeds = all_seeds[: args.seeds]
    print(f"[run_experiments] experimento={args.experiment} seeds={seeds} z={args.z}")
    print(f"[run_experiments] pso={params}")
    print(f"[run_experiments] resultados -> {args.out}")

    # Cenários (Parte 1) — sempre úteis como evidência.
    rows = runner.run_scenarios(save_path=args.out / "scenarios.csv")
    print(f"[run_experiments] cenários: {len(rows)} salvos em scenarios.csv")

    reports: dict = {}
    if args.experiment in ("calibration", "all"):
        print("[run_experiments] rodando calibração (E1)...")
        reports["calibracao"] = runner.run_calibration_experiment(seeds=seeds, params=params)
    if args.experiment in ("operational", "all"):
        print("[run_experiments] rodando busca operacional (E2)...")
        reports["operacional"] = runner.run_operational_experiment(
            z=args.z, seeds=seeds, params=params
        )

    runner.save_all(reports, args.out)
    for name, rep in reports.items():
        a = rep.aggregate
        print(f"[run_experiments] {name}: best={a.best:.3f} mean={a.mean:.3f} "
              f"std={a.std:.3f} (n={a.n_runs})")
    print(f"[run_experiments] [ok] resultados persistidos em {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
