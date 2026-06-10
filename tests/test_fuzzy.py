"""Testes do sistema fuzzy.

Portão A: valida a parte DECLARATIVA (base de regras e configuração), que não depende da lógica dos
algoritmos. A inferência fica marcada como `skip` até o Portão B.
"""
import pytest

from dfpo.config import MF_PARAMS, TEST_SCENARIOS, UNIVERSES
from dfpo.fuzzy.rules import RULE_TABLE


PORTAO_B_PENDENTE = "Lógica do Portão B ainda não implementada (stub)."


def test_rule_base_has_at_least_12_rules():
    assert len(RULE_TABLE) >= 12


def test_rule_base_covers_required_categories():
    tipos = {r.tipo for r in RULE_TABLE}
    for esperado in {"tipico", "intermediario", "fronteirico", "conflitante", "critico"}:
        assert esperado in tipos, f"categoria ausente na base de regras: {esperado}"


def test_rule_terms_are_valid():
    """Todo termo citado nas regras existe nas MFs configuradas."""
    for r in RULE_TABLE:
        assert r.x in MF_PARAMS["x"]
        assert r.y in MF_PARAMS["y"]
        assert r.z in MF_PARAMS["z"]
        assert r.s in MF_PARAMS["S"]


def test_at_least_six_scenarios():
    assert len(TEST_SCENARIOS) >= 6


def test_scenarios_within_universes():
    for s in TEST_SCENARIOS:
        assert UNIVERSES["x"][0] <= s.x <= UNIVERSES["x"][1]
        assert UNIVERSES["y"][0] <= s.y <= UNIVERSES["y"][1]
        assert UNIVERSES["z"][0] <= s.z <= UNIVERSES["z"][1]


def test_infer_returns_score_in_range():
    from dfpo.fuzzy.system import FuzzySystem
    fs = FuzzySystem().build()
    s = fs.infer(2.4, 2.0, 5.0)
    assert 0.0 <= s <= 100.0
