"""Testes do baseline analítico.

No Portão A a lógica ainda não existe -> testes marcados como `skip`. No Portão B, remover os skips:
o ponto crítico deve bater com (54/23, 60/23) ~= (2.348, 2.609), det(H)=23 e classificação de máximo
local estrito.
"""
import pytest

from dfpo.config import BASELINE_OPTIMUM


PORTAO_B_PENDENTE = "Lógica do Portão B ainda não implementada (stub)."


def test_critical_point_matches_specification():
    from dfpo.baseline.analytic import critical_point
    cp = critical_point()
    assert cp.x == pytest.approx(BASELINE_OPTIMUM["x"], abs=1e-3)   # ~2.348
    assert cp.y == pytest.approx(BASELINE_OPTIMUM["y"], abs=1e-3)   # ~2.609
    assert cp.det_hessian == pytest.approx(23.0, abs=1e-9)
    assert cp.f_xx == pytest.approx(-4.0, abs=1e-9)
    assert cp.classification == "max_local_estrito"


def test_gradient_zero_at_optimum():
    from dfpo.baseline.analytic import gradient
    gx, gy = gradient(BASELINE_OPTIMUM["x"], BASELINE_OPTIMUM["y"])
    assert gx == pytest.approx(0.0, abs=1e-6)
    assert gy == pytest.approx(0.0, abs=1e-6)


def test_config_optimum_constants_are_sane():
    """Sanidade dos CONSTANTES de configuração (não depende de lógica do Portão B)."""
    assert BASELINE_OPTIMUM["x"] == pytest.approx(2.3478, abs=1e-3)
    assert BASELINE_OPTIMUM["y"] == pytest.approx(2.6087, abs=1e-3)
