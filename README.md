# Drone Fuzzy-PSO Optimizer
### Sistema Inteligente Fuzzy-Evolutivo de Otimização de Asas para Drones de Entrega

- **Disciplina:** Inteligência Artificial e Computacional (0700M8) — Ciência da Computação, CESUPA
- **Professor:** Daniel Leal Souza · **Semestre:** 01/2026 · **Turma:** CC5NA
- **Equipe (4 integrantes):** `Carlos Eduardo Cardoso Silva`, `Gabriel Costa de Miranda`, `João Ricardo Silva de Almeida`, `Yago Patrick Schnorr Pinto`
- **Repositório:** `yagoschnorr/fuzzy-pso-wing-design`

## Modalidade (entrega única combinada)

Entrega **única** que satisfaz **simultaneamente** as duas laudas:

| Lauda | Modalidade |
|---|---|---|
| **Parte 1 — Sistemas de Controle Fuzzy** | Opção A (Pesquisa em artigos) + integração |
| **Parte 2 — IA Evolutiva e Comp. Bioinspirada** | Opção 1 (Pesquisa Científica), PSO |
| **Ponto extra** | Artigo de alto impacto (Qualis A1) adaptado |

**Artigo-base:** Kacimi, M.A.; Guenounou, O.; Brikh, L.; Yahiaoui, F.; Hadid, N. (2020).
*New mixed-coding PSO algorithm for a self-adaptive and automatic learning of Mamdani fuzzy rules.*
**Engineering Applications of Artificial Intelligence**, 89, 103417. Elsevier.
https://doi.org/10.1016/j.engappai.2019.103417

## Resumo da solução

Otimização de geometria de asa de drone de entrega via três pilares integrados:

1. **Baseline analítico** — superfície `f(x,y) = 12x + 18y − 2x² − 3y² − xy`, com gradiente, Hessiana
   e ótimo em `x≈2.348 m`, `y≈2.609 m` (máximo local estrito; `D=23>0`, `f_xx=−4<0`). Serve de
   referência exata para validar o fuzzy.
2. **Sistema fuzzy Mamdani** — entradas `x` (envergadura, [0,5] m), `y` (corda, [0,4] m),
   `z` (carga útil, [0,10] kg); saída `S` (score de eficiência, [0,100]%). ≥12 regras; operadores
   mínimo (conjunção), máximo (agregação) e centróide (defuzzificação). Implementado com **scikit-fuzzy**.
3. **PSO (Particle Swarm Optimization)** — implementado **manualmente**, em dois usos:
   - **Calibração** das funções de pertinência/pesos das regras minimizando o erro (MSE) contra uma
     referência (superfície analítica normalizada + dataset sintético para `z`) — núcleo científico,
     com comparação **antes/depois**.
   - **Busca operacional** — dado `z`, encontra `(x,y)` que maximiza `S(x,y,z)`.

## Tecnologias

Python 3.11 · NumPy · SciPy · **scikit-fuzzy** (fuzzy Mamdani) · Matplotlib · pandas ·
**FastAPI** (back-end, opcional) · **Streamlit** (dashboard) · PyYAML · pytest.

> Arquitetura em **camadas**: o núcleo `dfpo` não depende de FastAPI/Streamlit. Existe um
> **caminho mínimo viável só com Streamlit** (o dashboard cai para o núcleo direto se a API não estiver
> no ar — flag `USE_API`).

## Instalação

```bash
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Execução

```bash
# (Opcional) API FastAPI:
uvicorn api.main:app --reload --port 8000      # docs em http://localhost:8000/docs

# Caminho mínimo viável — dashboard roda sozinho (usa o núcleo se a API não estiver no ar):
streamlit run app/dashboard.py

# Experimentos: ≥5 sementes, métricas e curvas -> experiments/results/
python scripts/run_experiments.py --seeds 5 --config experiments/configs/default.yaml

# Figuras (superfície, sensibilidade, convergência) -> figures/
python scripts/gen_figures.py

# Tabelas consolidadas para o relatório -> experiments/results/report/
python scripts/make_report_assets.py
```

## Resultados (execução real, 5 sementes)

| Experimento | Resultado |
|---|---|
| **E1 — Calibração (PSO vs MSE)** | MSE **1584 → ~140** (média), melhora **~91%**; convergência ~iter 30 |
| **E2 — Busca operacional (z=5)** | **PSO S≈84,4 > busca aleatória ≈84,2 > busca gulosa ≈70,6**; ótimo robusto em (x≈2,5, y≈2,0) |
| **E3 — Cenários (≥6)** | saídas fuzzy coerentes (médio→Excelente, conflitante/crítico→Baixa) |

> Coerência verificada: ótimo fuzzy próximo do analítico `(2,348, 2,609)`; `MSE_depois < MSE_antes`;
> PSO supera as baselines. **Desempenho:** a calibração usa um avaliador Mamdani vetorizado
> (`src/dfpo/fuzzy/fast_eval.py`, ~135× mais rápido), com o scikit-fuzzy como sistema oficial de
> conferência (equivalência testada).

## Reprodução dos testes

```bash
pytest -q
```

> Regra de coerência: **todo número do relatório vem de `experiments/results/`** (saída de execução
> real). Nada é escrito à mão.

## Estrutura do repositório (principais arquivos)

```
src/dfpo/      núcleo científico (baseline, fuzzy, pso, reference, experiments, viz)
api/           FastAPI (/baseline, /fuzzy/infer, /pso/optimize, /pso/calibrate) — usa o núcleo real
app/           dashboard Streamlit
scripts/       run_experiments / gen_figures / make_report_assets
experiments/   configs/ (sementes, hiperparâmetros) + results/ (saídas)
docs/          manual de execução, base de regras, MFs, declaração de IA, esqueleto do artigo
tests/         testes (pytest)
PLAN.md        plano completo + matriz de rastreabilidade + cronograma
```

## Documentação

- [`docs/manual_execucao.md`](docs/manual_execucao.md) — manual de execução
- [`docs/base_de_regras.md`](docs/base_de_regras.md) — base de ≥12 regras
- [`docs/funcoes_pertinencia.md`](docs/funcoes_pertinencia.md) — funções de pertinência

## Licença

Ver [`LICENSE`](LICENSE).
