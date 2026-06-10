"""Verdade de referência para a calibração (estratégia híbrida confirmada).

Responsabilidade
----------------
Produzir os pares (entrada -> S_referencia) usados como alvo na calibração por PSO:

  (A) Superfície analítica: S_ref(x, y) = normaliza( f(x, y) ) para [0, 100], com z FIXO. Ancora a
      calibração na "verdade" exata do baseline no subespaço (x, y).
  (B) Dataset sintético/de especialista para a 3ª entrada (z): amostras (x, y, z -> S) que codificam o
      conhecimento de domínio sobre o efeito da carga útil (já que f não depende de z).

LIMITAÇÃO DECLARADA (honestidade metodológica): a função analítica f tem apenas 2 variáveis (x, y); o
efeito de z na referência vem do componente sintético/especialista, não de f. Isso é explicitado no
relatório.

ESTADO (Portão A): STUB.
"""
from __future__ import annotations

import numpy as np

from ..baseline import analytic
from ..config import UNIVERSES

# Resolução padrão da grade analítica. Mantida "coarse" (custo do scikit-fuzzy na conferência);
# o avaliador vetorizado da calibração lida bem com mais pontos, mas ~11x11 já guia o MSE.
DEFAULT_GRID_N = 11


def analytic_surface_reference(z_fixed: float, n: int = DEFAULT_GRID_N
                               ) -> tuple[np.ndarray, np.ndarray]:
    """Gera (entradas, S_ref) a partir de f(x, y) normalizada para [0,100], com z fixo.

    Returns
    -------
    inputs  : np.ndarray shape (n*n, 3)  -- colunas (x, y, z_fixed)
    targets : np.ndarray shape (n*n,)    -- S_ref em [0, 100]
    """
    lo_x, hi_x = UNIVERSES["x"]
    lo_y, hi_y = UNIVERSES["y"]
    xs = np.linspace(lo_x, hi_x, n)
    ys = np.linspace(lo_y, hi_y, n)
    X, Y = np.meshgrid(xs, ys)
    F = analytic.f(X, Y)
    S = 100.0 * (F - F.min()) / (F.max() - F.min())  # normalização min-max -> [0,100]

    inputs = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, float(z_fixed))])
    targets = S.ravel()
    return inputs, targets


# Pontos âncora de especialista codificando o efeito da carga útil z (coerentes com
# docs/base_de_regras.md). Formato: (x, y, z, S_esperado).
_PAYLOAD_ANCHORS: tuple[tuple[float, float, float, float], ...] = (
    # Geometria média: carga leve/padrão/pesada bem suportada -> Regular/Excelente.
    (2.4, 2.0, 2.0, 70.0),
    (2.4, 2.0, 5.0, 80.0),
    (2.4, 2.0, 8.0, 60.0),
    # Asa pequena: carga pesada inviável -> Baixa; carga leve -> Regular.
    (0.6, 0.5, 1.0, 55.0),
    (0.6, 0.5, 8.0, 12.0),
    (0.8, 0.6, 9.5, 8.0),
    # Asa grande: sustenta carga pesada -> Regular/Excelente; carga leve desperdiça -> Baixa.
    (3.8, 3.2, 9.0, 60.0),
    (4.2, 3.4, 9.5, 65.0),
    (3.8, 3.2, 1.0, 25.0),
    # Geometria intermediária com carga padrão.
    (2.0, 1.5, 5.0, 72.0),
    (3.0, 2.5, 5.0, 70.0),
    (1.5, 2.8, 5.0, 40.0),   # corda larga / envergadura curta -> conflitante
    (3.2, 1.0, 5.0, 38.0),   # envergadura longa / corda estreita -> conflitante
    (2.5, 1.6, 5.0, 78.0),   # ponto fronteiriço típico
)


def synthetic_payload_dataset(seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Dataset sintético/de especialista cobrindo as 3 entradas (com variação em z).

    Acrescenta ruído gaussiano leve (sigma=1.0, controlado por `seed`) aos alvos âncora.
    """
    rng = np.random.default_rng(seed)
    arr = np.array(_PAYLOAD_ANCHORS, dtype=float)
    inputs = arr[:, :3]
    targets = arr[:, 3] + rng.normal(0.0, 1.0, size=arr.shape[0])
    targets = np.clip(targets, 0.0, 100.0)
    return inputs, targets


def build_reference(z_fixed: float = 5.0, seed: int | None = None
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Combina (A) superfície analítica + (B) dataset sintético na referência final."""
    ia, ta = analytic_surface_reference(z_fixed)
    is_, ts = synthetic_payload_dataset(seed)
    return np.vstack([ia, is_]), np.concatenate([ta, ts])
