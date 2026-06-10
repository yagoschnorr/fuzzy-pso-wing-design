"""Base de regras do sistema fuzzy Mamdani (>= 12 regras).

Responsabilidade
----------------
Declarar a base de regras (ver tabela justificada em docs/base_de_regras.md) de forma DECLARATIVA
(dados) e materializá-la como objetos `skfuzzy.control.Rule` sobre as variáveis construídas em
`variables.py`.

A separação "declaração (dados) vs materialização (skfuzzy)" facilita:
  - documentar/auditar a cobertura (típicos, intermediários, fronteiriços, conflitantes, crítico);
  - aplicar pesos de regra otimizáveis pelo PSO na calibração (Portão B).

ESTADO (Portão A): a TABELA declarativa já está aqui (dados); a materialização em ctrl.Rule é STUB.
"""
from __future__ import annotations

from typing import NamedTuple

import skfuzzy.control as ctrl


class RuleSpec(NamedTuple):
    rid: str
    x: str          # termo de x  ('Curta'|'Media'|'Longa')
    y: str          # termo de y  ('Estreita'|'Media'|'Larga')
    z: str          # termo de z  ('Leve'|'Padrao'|'Pesada')
    s: str          # termo de S  ('Baixa'|'Regular'|'Excelente')
    tipo: str       # 'tipico'|'intermediario'|'fronteirico'|'conflitante'|'critico'
    weight: float = 1.0  # peso da regra (otimizável na calibração)


# Base declarativa — 16 regras (>= 12), espelha docs/base_de_regras.md.
RULE_TABLE: tuple[RuleSpec, ...] = (
    RuleSpec("R01", "Media", "Media",    "Padrao", "Excelente", "tipico"),
    RuleSpec("R02", "Curta", "Estreita", "Leve",   "Regular",   "tipico"),
    RuleSpec("R03", "Curta", "Estreita", "Pesada", "Baixa",     "critico"),
    RuleSpec("R04", "Longa", "Larga",    "Pesada", "Regular",   "tipico"),
    RuleSpec("R05", "Longa", "Larga",    "Leve",   "Baixa",     "conflitante"),
    RuleSpec("R06", "Media", "Media",    "Leve",   "Regular",   "intermediario"),
    RuleSpec("R07", "Media", "Media",    "Pesada", "Regular",   "intermediario"),
    RuleSpec("R08", "Media", "Estreita", "Padrao", "Regular",   "intermediario"),
    RuleSpec("R09", "Media", "Larga",    "Padrao", "Regular",   "intermediario"),
    RuleSpec("R10", "Curta", "Media",    "Padrao", "Baixa",     "fronteirico"),
    RuleSpec("R11", "Longa", "Media",    "Padrao", "Regular",   "fronteirico"),
    RuleSpec("R12", "Curta", "Larga",    "Padrao", "Baixa",     "conflitante"),
    RuleSpec("R13", "Longa", "Estreita", "Padrao", "Baixa",     "conflitante"),
    RuleSpec("R14", "Media", "Larga",    "Pesada", "Excelente", "tipico"),
    RuleSpec("R15", "Curta", "Estreita", "Padrao", "Baixa",     "fronteirico"),
    RuleSpec("R16", "Longa", "Larga",    "Padrao", "Regular",   "intermediario"),
)


def build_rules(variables, rule_table: tuple[RuleSpec, ...] = RULE_TABLE) -> list:
    """Materializa cada RuleSpec como skfuzzy.control.Rule.

    A conjunção dos antecedentes usa o operador `&` (mínimo, default do scikit-fuzzy).
    """
    regras = []
    for spec in rule_table:
        antecedente = variables.x[spec.x] & variables.y[spec.y] & variables.z[spec.z]
        regra = ctrl.Rule(antecedente, variables.S[spec.s], label=spec.rid)
        # spec.weight fica reservado para a calibração por pesos de regra (Portão B avançado).
        regras.append(regra)
    return regras
