"""Baseline analítico — superfície de referência exata.

Responsabilidade
----------------
Fornecer a função analítica e suas derivadas, localizar o ponto crítico e classificá-lo. Serve como
"verdade de referência" (subespaço x, y) para validar o sistema fuzzy e como alvo da calibração.

    f(x, y) = 12x + 18y - 2x^2 - 3y^2 - x*y

Resultados esperados (especificação, a serem REPRODUZIDOS pelo código no Portão B):
    grad f = [12 - 4x - y, 18 - 6y - x]
    Hessiana H = [[-4, -1], [-1, -6]]
    ponto crítico: x = 54/23 ~= 2.348 m, y = 60/23 ~= 2.609 m
    D = det(H) = (-4)(-6) - (-1)^2 = 23 > 0  e  f_xx = -4 < 0  => MÁXIMO LOCAL ESTRITO

ESTADO (Portão A): STUB. As funções abaixo ainda não estão implementadas.
"""
from __future__ import annotations

from typing import Literal, NamedTuple

import numpy as np


class CriticalPoint(NamedTuple):
    x: float
    y: float
    f_value: float
    det_hessian: float
    f_xx: float
    classification: Literal["max_local_estrito", "min_local_estrito", "sela", "indefinido"]


def f(x: float | np.ndarray, y: float | np.ndarray) -> float | np.ndarray:
    """Avalia f(x, y) = 12x + 18y - 2x^2 - 3y^2 - x*y. Aceita escalares ou arrays (vetorizado)."""
    return 12 * x + 18 * y - 2 * x**2 - 3 * y**2 - x * y


def gradient(x: float, y: float) -> tuple[float, float]:
    """Retorna o gradiente (df/dx, df/dy) no ponto (x, y)."""
    df_dx = 12 - 4 * x - y
    df_dy = 18 - 6 * y - x
    return (df_dx, df_dy)


def hessian() -> np.ndarray:
    """Retorna a Hessiana (constante): [[f_xx, f_xy], [f_xy, f_yy]] = [[-4, -1], [-1, -6]]."""
    return np.array([[-4.0, -1.0], [-1.0, -6.0]])


def critical_point() -> CriticalPoint:
    """Resolve grad f = 0, avalia f, calcula det(H) e classifica o ponto crítico.

    grad f = 0 equivale ao sistema linear:
        4x + y  = 12
        x  + 6y = 18
    cuja solução é (x*, y*) = (54/23, 60/23) ~= (2.348, 2.609).
    """
    # grad f = 0  ->  A [x, y]^T = b
    A = np.array([[4.0, 1.0], [1.0, 6.0]])
    b = np.array([12.0, 18.0])
    x_star, y_star = np.linalg.solve(A, b)

    H = hessian()
    det_h = float(np.linalg.det(H))
    f_xx = float(H[0, 0])

    if det_h > 0 and f_xx < 0:
        classification = "max_local_estrito"
    elif det_h > 0 and f_xx > 0:
        classification = "min_local_estrito"
    elif det_h < 0:
        classification = "sela"
    else:
        classification = "indefinido"

    return CriticalPoint(
        x=float(x_star),
        y=float(y_star),
        f_value=float(f(x_star, y_star)),
        det_hessian=det_h,
        f_xx=f_xx,
        classification=classification,
    )
