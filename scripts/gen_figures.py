#!/usr/bin/env python3
"""Gera as figuras de validação: superfície de controle, sensibilidade e convergência.

Interface (CLI) definida; geração real depende da lógica do Portão B.

Uso:
    python scripts/gen_figures.py --config experiments/configs/default.yaml --z 5.0
    python scripts/gen_figures.py --only surface
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

FIGS = ["surface", "sensitivity", "convergence"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gera as figuras de evidência do projeto.")
    p.add_argument("--only", choices=FIGS, default=None, help="Gera apenas uma figura.")
    p.add_argument("--z", type=float, default=5.0, help="Carga útil fixa p/ a superfície.")
    p.add_argument("--config", type=Path, default=ROOT / "experiments/configs/default.yaml")
    p.add_argument("--out", type=Path, default=ROOT / "figures", help="Diretório de saída das figuras.")
    return p.parse_args(argv)


def _load_convergence_histories(csv_path: Path) -> dict[str, list[float]]:
    """Lê um *_convergence.csv (formato largo) de volta para {série: histórico}."""
    import csv as _csv
    histories: dict[str, list[float]] = {}
    with csv_path.open(encoding="utf-8") as fh:
        reader = _csv.DictReader(fh)
        labels = [c for c in reader.fieldnames if c != "iteration"]
        for label in labels:
            histories[label] = []
        for row in reader:
            for label in labels:
                v = row[label]
                if v != "":
                    histories[label].append(float(v))
    return histories


def main(argv: list[str] | None = None) -> int:
    import matplotlib
    matplotlib.use("Agg")  # backend sem display (CI/headless)

    args = parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    targets = [args.only] if args.only else FIGS
    print(f"[gen_figures] gerando {targets} (z={args.z}) -> {args.out}")

    import numpy as np
    from dfpo.config import RESULTS_DIR
    from dfpo.fuzzy.system import FuzzySystem
    from dfpo.viz import convergence, sensitivity, surface

    fs = FuzzySystem().build()

    if "surface" in targets:
        surface.plot_control_surface(fs, z=args.z, n=40, save_path=args.out / "surface.png")
        print("[gen_figures] [ok] surface.png")

    if "sensitivity" in targets:
        sweep = np.linspace(1.6, 3.4, 12)
        sensitivity.sensitivity_curve(
            "x", "Media", 1, sweep, operating_point=(2.4, 2.0, args.z),
            save_path=args.out / "sensitivity.png",
        )
        print("[gen_figures] [ok] sensitivity.png")

    if "convergence" in targets:
        gerou = False
        for name in ("calibracao", "operacional"):
            csv_path = RESULTS_DIR / f"{name}_convergence.csv"
            if csv_path.exists():
                histories = _load_convergence_histories(csv_path)
                convergence.plot_convergence(
                    histories, save_path=args.out / f"convergence_{name}.png",
                    title=f"Convergência — {name}",
                )
                print(f"[gen_figures] [ok] convergence_{name}.png")
                gerou = True
        if not gerou:
            print("[gen_figures] (sem *_convergence.csv — rode scripts/run_experiments.py primeiro)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
