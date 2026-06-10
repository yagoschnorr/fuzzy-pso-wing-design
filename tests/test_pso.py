"""Testes do PSO.

Portão A: valida configuração de hiperparâmetros (dados). O loop do PSO, a busca operacional e a
calibração ficam marcados como `skip` até o Portão B.
"""
import pytest

from dfpo.config import DEFAULT_PSO, DEFAULT_SEEDS


PORTAO_B_PENDENTE = "Lógica do Portão B ainda não implementada (stub)."


def test_default_pso_hyperparameters():
    assert DEFAULT_PSO.n_particles == 30
    assert DEFAULT_PSO.n_iterations == 50
    assert DEFAULT_PSO.w == pytest.approx(0.729, abs=1e-3)
    assert DEFAULT_PSO.c1 == pytest.approx(1.494, abs=1e-3)
    assert DEFAULT_PSO.c2 == pytest.approx(1.494, abs=1e-3)


def test_at_least_five_seeds():
    assert len(DEFAULT_SEEDS) >= 5
    assert len(set(DEFAULT_SEEDS)) == len(DEFAULT_SEEDS)  # sementes distintas


def test_pso_minimizes_simple_quadratic():
    """Sanidade: PSO deve achar o mínimo de uma parábola simples."""
    import numpy as np
    from dfpo.pso.swarm import minimize
    res = minimize(lambda p: float(np.sum((p - 1.0) ** 2)), lb=[-5, -5], ub=[5, 5], seed=42)
    assert res.best_cost == pytest.approx(0.0, abs=1e-2)


def test_calibration_improves_mse():
    """Após a calibração, o MSE deve ser <= MSE inicial."""
    from dfpo.pso import calibrate
    from dfpo.reference.target import build_reference
    ref_in, ref_out = build_reference()
    res = calibrate.calibrate(ref_in, ref_out, seed=42)
    assert res.mse_after <= res.mse_before


def test_fast_evaluator_matches_scikit_fuzzy():
    """O avaliador Mamdani vetorizado (usado na calibração) deve concordar com o scikit-fuzzy."""
    from dfpo.config import TEST_SCENARIOS
    from dfpo.fuzzy.fast_eval import FastMamdani
    from dfpo.fuzzy.system import FuzzySystem
    fs = FuzzySystem().build()
    fm = FastMamdani()
    for s in TEST_SCENARIOS:
        official = fs.infer(s.x, s.y, s.z)
        fast = fm.infer(s.x, s.y, s.z)
        assert fast == pytest.approx(official, abs=0.5)  # escala S=0..100
