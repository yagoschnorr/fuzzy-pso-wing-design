# Portão B — Plano de Implementação (contexto para o Claude Code no repositório)

> **Como usar este arquivo:** cole-o como contexto no Claude Code aberto na raiz do seu clone
> `fuzzy-pso-wing-design` (já com o scaffolding do Portão A aplicado). Ele descreve, arquivo por
> arquivo, a lógica a implementar nos stubs. **Implemente na ordem dos marcos** (Parte 1 fuzzy antes de
> 11/06; Parte 2 PSO antes de 12/06). Não invente números: todo resultado do relatório deve vir da
> execução real (`experiments/results/`).

## Context

O repositório está no **Portão A**: estrutura, stubs (com assinaturas + pseudocódigo + `TODO(Portão B)`)
e mocks (`api/`, `app/`) já existem e foram validados (`pytest` = 7 passados, 5 *skipped*). O
**Portão B** consiste em **implementar a lógica real** dos algoritmos sem alterar as interfaces já
definidas, e fazer os mocks/endpoints/dashboard passarem a usar o núcleo real. Objetivo final: gerar os
**números reais** (calibração antes/depois, busca operacional, 6 cenários, sensibilidade, superfície,
convergência) que alimentam o PDF técnico e satisfazem as duas laudas.

Decisões fixadas pela equipe:
- **Dependências:** confirmadas como estão em `requirements.txt` (Python 3.11; `scikit-fuzzy==0.4.2`
  exige `numpy<2` — já pinado em `1.26.4`). Não mexer.
- **Funções de pertinência:** usar os defaults de `src/dfpo/config.py` (`MF_PARAMS`) como ponto de
  partida; não há valores preferidos.
- **Remover o CI:** apagar `.github/workflows/ci.yml` (não será usado por ora).
- **Fuzzy via `scikit-fuzzy`** (explicar o modelo no relatório); **PSO implementado manualmente**
  (sem `pyswarms`).

Princípio de coerência (rubrica das laudas): o sistema fuzzy deve ser consistente com o ótimo
analítico `(x≈2.348, y≈2.609)`; a calibração deve **reduzir o MSE** vs a referência; o PSO deve
**superar** a busca aleatória.

---

## Convenções gerais (válidas para todos os arquivos)
- **Não alterar assinaturas públicas nem nomes** já definidos nos stubs — só preencher o corpo.
- Substituir cada `raise NotImplementedError("Portão B: ...")` pela implementação.
- Usar `numpy` com gerador moderno: `rng = np.random.default_rng(seed)` (reprodutibilidade).
- Ao terminar cada módulo, **remover os `@pytest.mark.skip`** correspondentes em `tests/` e rodar
  `pytest -q` (todos devem passar).
- Salvar figuras em `figures/` e resultados numéricos em `experiments/results/`.

---

# MARCO 1 — Parte 1 (Fuzzy) — concluir até 11/06

### 1.1 `src/dfpo/baseline/analytic.py`
Implementar com base nas fórmulas (já documentadas no cabeçalho do arquivo):
- `f(x, y)` → `12*x + 18*y - 2*x**2 - 3*y**2 - x*y` (funciona vetorizado p/ arrays NumPy).
- `gradient(x, y)` → `(12 - 4*x - y, 18 - 6*y - x)`.
- `hessian()` → `np.array([[-4., -1.], [-1., -6.]])`.
- `critical_point()`:
  - resolver o sistema linear `4x + y = 12`, `x + 6y = 18` com `np.linalg.solve` →
    `(54/23, 60/23) ≈ (2.348, 2.609)`;
  - `D = np.linalg.det(hessian())` (= 23.0); `f_xx = -4.0`;
  - classificar: `D>0 e f_xx<0 → "max_local_estrito"`; `D>0 e f_xx>0 → "min_local_estrito"`;
    `D<0 → "sela"`; senão `"indefinido"`;
  - retornar `CriticalPoint(x, y, f(x,y), D, f_xx, classification)`.
- **Aceitação:** remover os 2 `skip` em `tests/test_baseline.py`; devem passar.

### 1.2 `src/dfpo/fuzzy/membership.py`
- `make_mf(universe, params)`: `len==3 → skfuzzy.trimf(universe, params)`;
  `len==4 → skfuzzy.trapmf(universe, params)`; senão `ValueError`.
- `is_valid_breakpoints(params, lo, hi)`: todos em `[lo,hi]` **e** não-decrescentes
  (`all(a<=b for a,b in zip(params, params[1:]))`).
- `import skfuzzy as fuzz` no topo.

### 1.3 `src/dfpo/fuzzy/variables.py`
- `build_variables(mf_params=None)`:
  - `params = mf_params or config.MF_PARAMS`;
  - construir o universo de cada entrada com passo fino, ex.: `np.arange(lo, hi + step, step)` com
    `step=0.01` para `x,y,z`; para `S` usar `config.S_RESOLUTION`;
  - `x = ctrl.Antecedent(universo_x, 'x')` (idem `y`, `z`); `S = ctrl.Consequent(universo_S, 'S')`;
  - para cada termo: `x['Curta'] = make_mf(x.universe, params['x']['Curta'])` etc.;
  - retornar `FuzzyVariables(x, y, z, S)`.
- `import numpy as np` e `import skfuzzy.control as ctrl`.

### 1.4 `src/dfpo/fuzzy/rules.py`
- `build_rules(variables, rule_table=RULE_TABLE)`: para cada `spec`,
  `ant = variables.x[spec.x] & variables.y[spec.y] & variables.z[spec.z]`;
  `regra = ctrl.Rule(ant, variables.S[spec.s])`; `regras.append(regra)`; retornar lista.
- (Pesos de regra `spec.weight` são opcionais; deixar comentado p/ possível uso na calibração.)

### 1.5 `src/dfpo/fuzzy/inference.py`
- `build_control_system(rules)` → `ctrl.ControlSystem(rules)`.
- `make_simulation(control_system)` → `ctrl.ControlSystemSimulation(control_system)`.
- `infer_once(simulation, x, y, z)`: setar `simulation.input['x'/'y'/'z']`, `simulation.compute()`,
  retornar `float(simulation.output['S'])`. Operadores mín/máx/centróide são o **default** do
  scikit-fuzzy (não precisa configurar; mencionar isso no relatório).

### 1.6 `src/dfpo/fuzzy/system.py`
- `build()`: `v = variables.build_variables(self.mf_params)`; `r = rules.build_rules(v, self.rule_table)`;
  `cs = inference.build_control_system(r)`; guardar `self._cs = cs`; `self._sim = inference.make_simulation(cs)`;
  `return self`.
- `infer(x, y, z)`: se `self._sim is None: self.build()`. **Importante (estabilidade do scikit-fuzzy):**
  o `ControlSystemSimulation` acumula estado em chamadas repetidas; para robustez, **criar uma
  simulação nova por chamada** a partir de `self._cs` (`inference.make_simulation(self._cs)`) ou
  recriá-la a cada N chamadas. Retornar o `S`.
- `infer_grid(z, n=50)`: `xs = linspace(universo x, n)`, `ys = linspace(universo y, n)`,
  `X,Y = meshgrid`; preencher `S[i,j] = self.infer(X[i,j], Y[i,j], z)`; retornar `(X, Y, S)`.
- **Aceitação:** remover o `skip` de `test_infer_returns_score_in_range` em `tests/test_fuzzy.py`.

### 1.7 `src/dfpo/viz/surface.py` e `src/dfpo/viz/sensitivity.py`
- `plot_control_surface(fs, z, n, save_path)`: `X,Y,S = fs.infer_grid(z,n)`; superfície 3D
  (`mpl_toolkits.mplot3d`), rótulos `x [m]`, `y [m]`, `S [%]`, título com `z`; salvar e retornar `(fig, ax)`.
- `sensitivity_curve(variable, term, param_index, sweep_values, operating_point, save_path)`:
  para cada `v` em `sweep_values`, copiar `config.MF_PARAMS` (deepcopy), setar
  `mf[variable][term][param_index]=v`, `fs=FuzzySystem(mf_params=mf).build()`,
  coletar `fs.infer(*operating_point)`; plotar `S` vs `sweep_values`; retornar `(sweep_values, S_out, fig)`.

### 1.8 Camada de apresentação da Parte 1 (tirar do mock)
- `api/main.py`: em `/baseline`, usar `dfpo.baseline.analytic.critical_point()` e preencher valores
  reais, `placeholder=False`; em `/fuzzy/infer`, construir um `FuzzySystem` (cachear em nível de módulo)
  e retornar `fs.infer(...)`, `placeholder=False`.
- `app/dashboard.py`: trocar os `TODO(Portão B)` de `get_baseline` e `get_fuzzy_score` pelas chamadas
  reais ao núcleo (quando `USE_API=0`).

### 1.9 Cenários (≥6) e evidências da Parte 1
- Em `experiments/runner.py` (ou um script simples), avaliar `config.TEST_SCENARIOS` com o
  `FuzzySystem` real e salvar tabela em `experiments/results/scenarios.csv` (entradas, `S`,
  interpretação). Gerar `figures/surface.png` e `figures/sensitivity.png`.

---

# MARCO 2 — Parte 2 (Evolutiva / PSO) — concluir até 12/06

### 2.1 `src/dfpo/pso/swarm.py` — PSO genérico (manual)
Implementar `minimize(objective, lb, ub, params, seed)` exatamente como o pseudocódigo do stub:
- inicializar posições `X` uniformes em `[lb,ub]`, velocidades em `[-v_max, v_max]` com
  `v_max = params.v_max_frac*(ub-lb)`;
- `pbest`, `gbest`; atualização com inércia/constrição:
  `V = w*V + c1*r1*(pbest-X) + c2*r2*(gbest-X)`; clampar `V` e `X`;
- registrar `history` (gbest por iteração), `n_evaluations` (nfe), `convergence_iter`
  (via `metrics.detect_convergence(history)`), `elapsed_s` (`time.perf_counter`);
- retornar `PSOResult`.
- **Aceitação:** remover `skip` de `test_pso_minimizes_simple_quadratic` (parábola → custo ≈ 0).

### 2.2 `src/dfpo/experiments/metrics.py`
- `detect_convergence(history, tol, patience)`: 1ª iteração a partir da qual a melhora absoluta do
  gbest fica `< tol` por `patience` iterações consecutivas (senão `None`).
- `aggregate(runs)`: `best/worst/mean/std` de `best_cost` (NumPy) + médias de `convergence_iter`,
  `elapsed_s`, `n_evaluations`; retornar `AggregateMetrics`.

### 2.3 `src/dfpo/experiments/baselines.py`
- `random_search(objective, lb, ub, n_evaluations, seed)`: amostrar `n_evaluations` pontos uniformes,
  avaliar, acompanhar o melhor; retornar `(best_pos, best_cost, history)`.
- `greedy_search(...)`: hill-climbing com passo `step_frac*(ub-lb)` e perturbação gaussiana; aceitar
  vizinho se melhora; respeitar orçamento de avaliações.

### 2.4 `src/dfpo/pso/operate.py` — busca operacional
- `optimize_geometry(fs, z, params, seed)`: `custo(p) = -fs.infer(p[0], p[1], z)`;
  `lb=[0,0]`, `ub=[5,4]` (de `config.UNIVERSES`); `res = swarm.minimize(...)`;
  retornar `OperationResult(x, y, -res.best_cost, res)`.
- Reusar **um único** `FuzzySystem` já `build()`-ado (MFs fixas) → barato.

### 2.5 `src/dfpo/reference/target.py` — verdade de referência (híbrida)
- `analytic_surface_reference(z_fixed, n)`: `X,Y` em meshgrid; `F = baseline.analytic.f(X,Y)`;
  normalizar min-max para `[0,100]`: `S = 100*(F-F.min())/(F.max()-F.min())`; montar `inputs (n*n,3)`
  com coluna `z=z_fixed` e `targets` achatados.
- `synthetic_payload_dataset(seed)`: ~10–20 pontos âncora `(x,y,z→S)` codificando o efeito de `z`
  (coerentes com `docs/base_de_regras.md`; ex.: carga pesada com asa pequena → S baixo). Opcional:
  ruído leve controlado por `rng`.
- `build_reference(z_fixed=5.0, seed)`: empilhar (A)+(B) → `(inputs, targets)`.

### 2.6 `src/dfpo/pso/calibrate.py` — NÚCLEO CIENTÍFICO (adaptação de Kacimi et al., 2020)
- `encode(mf_params)`: achatar, em ordem fixa e documentada, os breakpoints otimizáveis das MFs em um
  vetor. **Recomendação inicial (real-coded):** otimizar os pontos **interiores** das MFs de entrada e
  saída; manter fixos os pontos colados às bordas dos universos (0/5, 0/4, 0/10, 0/100) para reduzir a
  dimensionalidade.
- `decode(vector, template)`: reconstruir o dicionário e **reparar restrições** — ordenar breakpoints
  (não-decrescente) e clipar aos limites do universo (usar `membership.is_valid_breakpoints` para
  checagem).
- `make_objective(reference_inputs, reference_outputs, template)`: retornar `objective(vector)` que
  `decode → FuzzySystem(mf_params=mf).build() → infer` em cada linha de `reference_inputs` →
  `MSE(preds, reference_outputs)`.
- `calibrate(...)`: `mse_before = objective(encode(template))`; `res = swarm.minimize(objective, lb, ub,
  params, seed)`; `mf_after = decode(res.best_position, template)`; `mse_after = res.best_cost`;
  `improvement = 100*(mse_before-mse_after)/mse_before`; retornar `CalibrationResult`.
- **Extensão opcional (ponto extra reforçado — mixed-coding do artigo):** otimizar também as
  **conclusões inteiras** das regras (`RuleSpec.s`) via *monitoring function* + limiar auto-adaptativo
  (Eqs. 3–5 do artigo). Implementar só depois que a versão real-coded estiver funcionando.
- **Aceitação:** remover `skip` de `test_calibration_improves_mse` (`mse_after <= mse_before`).

### 2.7 `src/dfpo/experiments/runner.py`
- `run_calibration_experiment(seeds, params)`: `ref_in, ref_out = reference.build_reference()`; para
  cada `seed`, `res = calibrate.calibrate(ref_in, ref_out, params=params, seed=seed)`; montar
  `RunMetrics`; `agg = metrics.aggregate(runs)`; `extra = {mse_before, mse_after, improvement_pct,
  histories}`; retornar `ExperimentReport('calibracao', agg, extra)`.
- `run_operational_experiment(z, seeds, params)`: `fs = FuzzySystem().build()`; para cada `seed`,
  `operate.optimize_geometry(fs, z, params, seed)` + rodar `baselines.random_search`/`greedy_search`
  com **mesmo nfe** para comparação; agregar; `extra` com baselines e curvas.
- `run_all(seeds, params)`: chamar os dois acima (p/ `z` em `config`), retornar dict de relatórios.

### 2.8 `src/dfpo/viz/convergence.py`
- `plot_convergence(histories, save_path, title)`: plotar cada `history` (PSO por semente + baselines)
  como linhas; eixos `Iteração`/`Melhor custo`; legenda; salvar; retornar `(fig, ax)`.

### 2.9 Scripts (ligar interface → núcleo) e persistência
- `scripts/run_experiments.py`: trocar o bloco `try/NotImplementedError` por: carregar
  `experiments/configs/default.yaml` (PyYAML), chamar `runner.run_all(...)`, e **persistir** em
  `experiments/results/` (JSON + CSV): métricas agregadas, MSE antes/depois, históricos de convergência.
- `scripts/gen_figures.py`: chamar `viz.surface`, `viz.sensitivity`, `viz.convergence` salvando em
  `figures/`.
- `scripts/make_report_assets.py`: ler `experiments/results/*` com pandas e montar tabelas
  markdown/CSV consolidadas (qualidade, convergência, custo, antes/depois) para o PDF.

### 2.10 Camada de apresentação da Parte 2 (tirar do mock)
- `api/main.py`: `/pso/optimize` → `operate.optimize_geometry`; `/pso/calibrate` →
  `runner.run_calibration_experiment` (resumo); `placeholder=False`.
- `app/dashboard.py`: aba **Busca Operacional** chama o PSO real; aba **Calibração** mostra MSE
  antes/depois + curva real (ou lê de `experiments/results/`).

### 2.11 Limpeza
- **Apagar** `.github/workflows/ci.yml` (decisão da equipe).

---

## Desempenho (atenção — risco real do scikit-fuzzy na calibração)
Cada avaliação da função objetivo da calibração **reconstrói** um `ControlSystem` (as MFs mudam) e
avalia `|referência|` pontos. Com 30 partículas × 50 iterações × 5 sementes isso multiplica rápido.
Mitigações recomendadas (todas configuráveis):
1. **Referência coarse:** usar `n≈9–11` na `analytic_surface_reference` (≈81–121 pontos) + ~15
   sintéticos. Suficiente para guiar o MSE.
2. **Iterações da calibração:** começar com `n_iterations≈30` e subir se o tempo permitir (manter 50
   para a busca operacional, que é barata).
3. **Reaproveitar** o máximo possível na construção das variáveis; evitar `deepcopy` desnecessário.
4. (Opcional avançado) Implementar um avaliador Mamdani **vetorizado em NumPy** só para o laço interno
   da calibração (mín/máx/centróide), mantendo o `scikit-fuzzy` como sistema "oficial"/conferência —
   acelera ordens de magnitude. Fazer só se o tempo de execução inviabilizar as 5 sementes.
5. Rodar sementes em paralelo (`concurrent.futures.ProcessPoolExecutor`) se necessário.

Documentar no relatório o `n` da referência, iterações e tempos efetivamente usados (custo
computacional é exigido pela Parte 2).

---

## Ordem sugerida de execução (resumo)
1. `baseline/analytic.py` → un-skip testes de baseline.
2. `fuzzy/`: membership → variables → rules → inference → system → un-skip teste de inferência.
3. `viz/surface.py` + `viz/sensitivity.py`; gerar 6 cenários + figuras (fecha **Parte 1**).
4. `pso/swarm.py` → un-skip teste do PSO; `metrics.py`; `baselines.py`.
5. `pso/operate.py` (busca operacional).
6. `reference/target.py` → `pso/calibrate.py` → un-skip teste de calibração.
7. `experiments/runner.py` + `viz/convergence.py`; ligar `scripts/*`; rodar ≥5 sementes; salvar
   resultados/figuras (fecha **Parte 2**).
8. Tirar `api/` e `app/` do mock (`placeholder=False`); apagar CI.

## Verificação end-to-end (antes de entregar)
- `pytest -q` → **todos passam** (sem `skip`); incluir os asserts de baseline, ≥12 regras, ≥6
  cenários, ≥5 sementes, hiperparâmetros, calibração reduz MSE.
- `python scripts/run_experiments.py --seeds 5` → gera arquivos em `experiments/results/`.
- `python scripts/gen_figures.py` → gera `figures/surface.png`, `sensitivity.png`, `convergence.png`.
- `python scripts/make_report_assets.py` → tabelas consolidadas.
- `uvicorn api.main:app` → endpoints retornam dados reais com `placeholder=False`.
- `streamlit run app/dashboard.py` → 4 abas com números reais.
- Conferir coerência: ótimo fuzzy próximo de `(2.348, 2.609)`; `mse_after < mse_before`;
  PSO ≥ busca aleatória. Atualizar `docs/AI_USAGE_DECLARATION.md` com o uso de IA do Portão B.
