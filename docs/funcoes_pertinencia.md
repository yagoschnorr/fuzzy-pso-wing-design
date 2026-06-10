# Funções de Pertinência (MFs) — ponto de partida

> ⚠️ **Portão A:** os parâmetros abaixo são o **ponto de partida** definido a partir dos universos da
> especificação. Eles serão **calibrados pelo PSO** no Portão B (a versão calibrada será documentada ao
> lado da inicial, para a comparação antes/depois). Valores numéricos finais virão da execução real.

Operadores do sistema Mamdani: **conjunção = mínimo**, **agregação = máximo**,
**defuzzificação = centróide**. Formas: **triangular** (`trimf`) e **trapezoidal** (`trapmf`),
via `scikit-fuzzy`.

## Entradas

### x — Envergadura `[0, 5] m` (3 termos)
| Termo | Forma | Parâmetros (m) |
|---|---|---|
| Curta   | trapezoidal | `[0.0, 0.0, 1.0, 2.0]` |
| Média   | triangular  | `[1.5, 2.5, 3.5]` |
| Longa   | trapezoidal | `[3.0, 4.0, 5.0, 5.0]` |

### y — Corda `[0, 4] m` (3 termos)
| Termo | Forma | Parâmetros (m) |
|---|---|---|
| Estreita | trapezoidal | `[0.0, 0.0, 0.8, 1.6]` |
| Média    | triangular  | `[1.2, 2.0, 2.8]` |
| Larga    | trapezoidal | `[2.4, 3.2, 4.0, 4.0]` |

### z — Carga útil `[0, 10] kg` (3 termos)
| Termo | Forma | Parâmetros (kg) |
|---|---|---|
| Leve    | trapezoidal | `[0.0, 0.0, 2.0, 4.0]` |
| Padrão  | triangular  | `[3.0, 5.0, 7.0]` |
| Pesada  | trapezoidal | `[6.0, 8.0, 10.0, 10.0]` |

## Saída

### S — Score de eficiência `[0, 100] %` (3 termos)
| Termo | Forma | Parâmetros (%) |
|---|---|---|
| Baixa     | trapezoidal | `[0, 0, 20, 40]` |
| Regular   | triangular  | `[30, 50, 70]` |
| Excelente | trapezoidal | `[60, 80, 100, 100]` |

## Restrições para a calibração (Portão B)
- **Ordenação dos breakpoints:** dentro de cada MF, `p1 ≤ p2 ≤ p3 (≤ p4)`.
- **Limites:** todos os pontos dentro do respectivo universo de discurso.
- **Cobertura:** os termos devem cobrir o universo sem deixar buracos e sem se anular mutuamente
  (premissa de interpretabilidade semântica, inspirada no artigo-base).

## Evidências a gerar (Portão B)
- Gráfico de cada conjunto de MFs (antes e depois da calibração).
- Curva de **sensibilidade** de ≥1 MF (ex.: variar o pico de `x:Média` e medir efeito em `S`).
- **Superfície de controle 3D** `S(x, y)` com `z` fixo.
