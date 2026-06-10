"""Dashboard Streamlit — Drone Fuzzy-PSO Optimizer.

Caminho mínimo viável: este app funciona SOZINHO (sem FastAPI). Se a flag de ambiente USE_API=1
estiver setada e a API estiver no ar, ele consome a API; caso contrário, usa o núcleo `dfpo`
diretamente.

ESTADO (Portão B): todas as abas usam o núcleo REAL (inferência fuzzy, PSO e calibração).

Rodar:  streamlit run app/dashboard.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import streamlit as st

# Garante que `dfpo` (em src/) seja importável quando rodando sem instalar o pacote.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

USE_API = os.environ.get("USE_API", "0") == "1"
API_URL = os.environ.get("API_URL", "http://localhost:8000")
RESULTS_DIR = ROOT / "experiments" / "results"


@st.cache_resource
def _fuzzy_system():
    """Constrói o FuzzySystem uma única vez (caro) e reutiliza entre interações."""
    from dfpo.fuzzy.system import FuzzySystem
    return FuzzySystem().build()


# --------------------------------------------------------------------------- #
# Camada de dados: API (se USE_API) ou núcleo direto.
# --------------------------------------------------------------------------- #
def get_baseline() -> dict:
    if USE_API:
        try:
            import requests
            return requests.get(f"{API_URL}/baseline", timeout=3).json()
        except Exception:
            st.warning("API indisponível — usando o núcleo local.")
    from dfpo.baseline.analytic import critical_point
    cp = critical_point()
    return {
        "formula": "f(x,y) = 12x + 18y - 2x^2 - 3y^2 - x*y",
        "optimum_x": cp.x, "optimum_y": cp.y, "f_optimum": cp.f_value,
        "det_hessian": cp.det_hessian, "f_xx": cp.f_xx,
        "classification": cp.classification, "placeholder": False,
    }


def get_fuzzy_score(x: float, y: float, z: float) -> dict:
    if USE_API:
        try:
            import requests
            return requests.post(f"{API_URL}/fuzzy/infer",
                                 json={"x": x, "y": y, "z": z}, timeout=3).json()
        except Exception:
            st.warning("API indisponível — usando o núcleo local.")
    score = _fuzzy_system().infer(x, y, z)
    return {"score": score, "inputs": {"x": x, "y": y, "z": z}, "placeholder": False}


def get_pso_optimize(z: float, seed: int = 42) -> dict:
    if USE_API:
        try:
            import requests
            return requests.post(f"{API_URL}/pso/optimize",
                                 json={"z": z, "seed": seed}, timeout=30).json()
        except Exception:
            st.warning("API indisponível — usando o núcleo local.")
    from dfpo.config import DEFAULT_PSO
    from dfpo.pso import operate
    res = operate.optimize_geometry(_fuzzy_system(), z, params=DEFAULT_PSO, seed=seed)
    return {"best_x": res.x, "best_y": res.y, "best_score": res.score,
            "convergence": [-c for c in res.pso.history], "placeholder": False}


def get_calibration_from_results() -> dict | None:
    """Lê o último resultado de calibração persistido (se houver)."""
    metrics = RESULTS_DIR / "calibracao_metrics.json"
    if not metrics.exists():
        return None
    data = json.loads(metrics.read_text(encoding="utf-8"))
    return data.get("extra")


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="Drone Fuzzy-PSO Optimizer", layout="wide")
st.title("🛩️ Drone Fuzzy-PSO Optimizer")
st.caption("Sistema Inteligente Fuzzy-Evolutivo de Otimização de Asas para Drones de Entrega — CC5NA")
st.caption(f"Fonte de dados: {'API FastAPI' if USE_API else 'núcleo local (dfpo)'} "
           f"({'USE_API=1' if USE_API else 'USE_API=0'})")

tab_base, tab_fuzzy, tab_oper, tab_calib = st.tabs(
    ["📐 Baseline", "🔢 Inferência Fuzzy", "🎯 Busca Operacional", "🛠️ Calibração (PSO)"]
)

with tab_base:
    st.subheader("Baseline analítico")
    b = get_baseline()
    st.latex(r"f(x,y) = 12x + 18y - 2x^2 - 3y^2 - xy")
    c1, c2, c3 = st.columns(3)
    c1.metric("Ótimo x [m]", f"{b['optimum_x']:.3f}")
    c2.metric("Ótimo y [m]", f"{b['optimum_y']:.3f}")
    c3.metric("det(H)", f"{b['det_hessian']:.0f}")
    st.write(f"Classificação: **{b['classification']}** (f_xx = {b['f_xx']}).")

with tab_fuzzy:
    st.subheader("Inferência fuzzy S(x, y, z)")
    x = st.slider("x — envergadura [m]", 0.0, 5.0, 2.4, 0.1)
    y = st.slider("y — corda [m]", 0.0, 4.0, 2.0, 0.1)
    z = st.slider("z — carga útil [kg]", 0.0, 10.0, 5.0, 0.1)
    r = get_fuzzy_score(x, y, z)
    st.metric("Score de eficiência S [%]", f"{r['score']:.1f}")
    st.caption("Mamdani (mín/máx/centróide) via scikit-fuzzy.")

with tab_oper:
    st.subheader("Busca operacional: dado z, maximizar S(x, y)")
    z_op = st.slider("z — carga útil [kg] ", 0.0, 10.0, 5.0, 0.1, key="z_op")
    if st.button("Otimizar geometria (PSO)"):
        with st.spinner("Rodando PSO..."):
            r = get_pso_optimize(z_op)
        c1, c2, c3 = st.columns(3)
        c1.metric("x* [m]", f"{r['best_x']:.3f}")
        c2.metric("y* [m]", f"{r['best_y']:.3f}")
        c3.metric("S* [%]", f"{r['best_score']:.1f}")
        st.line_chart({"melhor S (convergência)": r["convergence"]})

with tab_calib:
    st.subheader("Calibração das MFs por PSO (antes/depois)")
    st.write("Núcleo científico — adaptação de Kacimi et al. (2020), *EAAI* (mixed-coding PSO).")

    cached = get_calibration_from_results()
    if cached:
        st.success("Resultado lido de `experiments/results/calibracao_metrics.json`.")
        c1, c2, c3 = st.columns(3)
        c1.metric("MSE antes", f"{cached['mse_before']:.2f}")
        c2.metric("MSE depois (média)", f"{cached['mse_after_mean']:.2f}")
        c3.metric("Melhora", f"{cached['improvement_pct_mean']:.1f}%")
        hist = cached.get("histories", {})
        if hist:
            st.line_chart({k: v for k, v in hist.items()})
    else:
        st.info("Sem resultado persistido ainda. Rode `python scripts/run_experiments.py --seeds 5` "
                "ou execute uma calibração rápida abaixo.")

    if st.button("Rodar calibração rápida (1 semente, 30 iterações)"):
        from dfpo.config import PSOParams
        from dfpo.pso import calibrate
        from dfpo.reference.target import build_reference
        with st.spinner("Calibrando..."):
            ref_in, ref_out = build_reference()
            res = calibrate.calibrate(ref_in, ref_out,
                                      params=PSOParams(n_iterations=30), seed=42)
        c1, c2, c3 = st.columns(3)
        c1.metric("MSE antes", f"{res.mse_before:.2f}")
        c2.metric("MSE depois", f"{res.mse_after:.2f}")
        c3.metric("Melhora", f"{res.improvement_pct:.1f}%")
        st.line_chart({"PSO (MSE)": list(res.pso.history)})
