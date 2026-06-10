# Manual de Execução

> Estado atual: **Portão B (implementado)**. Todos os comandos abaixo produzem **resultados reais**
> a partir do núcleo `dfpo` (sem mocks). A suíte `pytest` passa integralmente.

## 1. Pré-requisitos
- Python **3.11**
- `pip` e `venv`
- (Opcional) `git`

## 2. Criar ambiente e instalar dependências

```bash
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

(Opcional, para `import dfpo` sem ajustar PATH) instalar o pacote em modo editável:
```bash
pip install -e .
```

## 3. Caminho mínimo viável — só Streamlit

O dashboard funciona **sem** o back-end. Se a API não estiver no ar, ele usa o núcleo `dfpo`
diretamente.

```bash
streamlit run app/dashboard.py
```

Abra o endereço exibido (normalmente http://localhost:8501).

## 4. Back-end FastAPI (opcional)

```bash
uvicorn api.main:app --reload --port 8000
```

- Documentação interativa: http://localhost:8000/docs
- Endpoints: `GET /baseline`, `POST /fuzzy/infer`, `POST /pso/optimize`, `POST /pso/calibrate`

Para o dashboard consumir a API, defina a flag antes de subir o Streamlit:
```bash
export USE_API=1          # Windows (PowerShell): $env:USE_API=1
export API_URL=http://localhost:8000
streamlit run app/dashboard.py
```

## 5. Experimentos

```bash
# Executa o protocolo experimental com >= 5 sementes independentes
python scripts/run_experiments.py --seeds 5 --config experiments/configs/default.yaml

# Gera figuras (superfície de controle, sensibilidade, convergência)
python scripts/gen_figures.py --config experiments/configs/default.yaml

# Consolida os números reais para o relatório
python scripts/make_report_assets.py
```

Saídas:
- `experiments/results/` — `*_metrics.json`, `*_runs.csv`, `*_convergence.csv`, `scenarios.csv`
- `experiments/results/report/resultados_consolidados.md` — tabelas E1/E2/E3 para o relatório
- `figures/` — `surface.png`, `sensitivity.png`, `convergence_calibracao.png`, `convergence_operacional.png`

> A ordem importa: `gen_figures.py` lê as curvas de convergência geradas por `run_experiments.py`.

## 6. Testes

```bash
pytest -q
```

> Todos os testes passam (baseline, ≥12 regras, ≥6 cenários, ≥5 sementes, hiperparâmetros, PSO
> minimiza, calibração reduz MSE e equivalência do avaliador vetorizado vs scikit-fuzzy).

## 7. Reprodutibilidade
- Versões fixadas em `requirements.txt`.
- Sementes e hiperparâmetros em `experiments/configs/default.yaml`.
- **Regra de coerência:** todo número do relatório vem de `experiments/results/` (execução real).
