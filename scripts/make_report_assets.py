#!/usr/bin/env python3
"""Consolida os números reais (de experiments/results/) em tabelas para o relatório.

Garante a REGRA DE COERÊNCIA das laudas: todo número do relatório vem da execução real, não escrito à
mão. Interface definida; consolidação real depende do Portão B (quando houver resultados).

Uso:
    python scripts/make_report_assets.py --results experiments/results --out experiments/results/report
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dfpo.config import RESULTS_DIR  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Consolida resultados em tabelas para o relatório.")
    p.add_argument("--results", type=Path, default=RESULTS_DIR, help="Pasta com resultados brutos.")
    p.add_argument("--out", type=Path, default=RESULTS_DIR / "report", help="Saída consolidada.")
    return p.parse_args(argv)


def _load_json(path: Path) -> dict:
    import json
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _df_to_md(df, round_to: int = 3) -> str:
    """Converte um DataFrame em tabela markdown (sem depender de `tabulate`)."""
    df = df.round(round_to)
    cols = list(df.columns)
    head = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *rows])


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"[make_report_assets] lendo {args.results} -> consolidando em {args.out}")

    import pandas as pd

    raw = sorted(args.results.glob("*"))
    if not raw:
        print("[make_report_assets] (nenhum resultado — rode scripts/run_experiments.py primeiro)")
        return 0

    md: list[str] = ["# Resultados consolidados — Drone Fuzzy-PSO Optimizer\n",
                     "> Gerado por `scripts/make_report_assets.py` a partir de `experiments/results/`.",
                     "> Regra de coerência: todo número do relatório vem destes arquivos.\n"]

    # 1) Calibração (E1) — qualidade + antes/depois + custo.
    cal = _load_json(args.results / "calibracao_metrics.json")
    if cal:
        agg, extra = cal["aggregate"], cal["extra"]
        md.append("## E1 — Calibração do fuzzy por PSO (MSE)\n")
        tbl = pd.DataFrame([{
            "MSE antes": round(extra["mse_before"], 3),
            "MSE depois (média)": round(extra["mse_after_mean"], 3),
            "Melhora (%)": round(extra["improvement_pct_mean"], 2),
            "MSE best": round(agg["best"], 3),
            "MSE worst": round(agg["worst"], 3),
            "MSE std": round(agg["std"], 3),
            "Conv. iter (média)": agg["mean_convergence_iter"],
            "Tempo (s, média)": round(agg["mean_elapsed_s"], 3),
            "nfe (média)": agg["mean_evaluations"],
            "Pontos referência": extra["n_reference_points"],
        }])
        md.append(_df_to_md(tbl) + "\n")
        runs = args.results / "calibracao_runs.csv"
        if runs.exists():
            md.append("**Por semente:**\n")
            md.append(_df_to_md(pd.read_csv(runs)) + "\n")

    # 2) Operacional (E2) — PSO vs baselines + geometrias.
    op = _load_json(args.results / "operacional_metrics.json")
    if op:
        extra = op["extra"]
        md.append("## E2 — Busca operacional (PSO vs baselines)\n")
        tbl = pd.DataFrame([{
            "z (kg)": extra["z"],
            "PSO S (média)": round(extra["pso_best_S_mean"], 3),
            "Busca aleatória S (média)": round(extra["random_best_S_mean"], 3),
            "Busca gulosa S (média)": round(extra["greedy_best_S_mean"], 3),
        }])
        md.append(_df_to_md(tbl) + "\n")
        geo = pd.DataFrame(extra["best_geometries"])
        md.append("**Melhores geometrias (x, y, S) por semente:**\n")
        md.append(_df_to_md(geo) + "\n")

    # 3) Cenários (E3).
    scen = args.results / "scenarios.csv"
    if scen.exists():
        md.append("## E3 — Cenários de teste (saída fuzzy real)\n")
        md.append(_df_to_md(pd.read_csv(scen)) + "\n")

    out_md = args.out / "resultados_consolidados.md"
    out_md.write_text("\n".join(md), encoding="utf-8")
    print(f"[make_report_assets] [ok] {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
