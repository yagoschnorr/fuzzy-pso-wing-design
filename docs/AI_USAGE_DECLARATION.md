# Declaração de Uso de IA

> Exigência das duas laudas (Parte 1 e Parte 2). Este documento é um **registro vivo**: deve ser
> atualizado pela equipe à medida que a IA for usada, sempre com **revisão humana** declarada.
> Declarar o uso **não** reduz a nota; o que reduz é usar IA sem compreender, revisar, validar, citar
> ou declarar.

## Resumo
A equipe utilizou assistência de IA para acelerar a **estruturação** do projeto e a redação de
documentação, mantendo a responsabilidade integral sobre o conteúdo final, com revisão, execução e
validação humanas.

## Registro de usos

| Ferramenta | Finalidade | Prompt/comando resumido | Revisão crítica da equipe |
|---|---|---|---|
| Claude Code (claude.ai/code) | Gerar **esqueleto do repositório** (estrutura de pastas, stubs com assinaturas + pseudocódigo, `TODO`), `requirements.txt`, `README.md`, manual de execução, especificação das MFs e base de regras, **endpoints FastAPI mock** e **dashboard Streamlit mock**, scripts de experimento (stubs). | "Planejar e andaimar (sem implementar a lógica final) o projeto Fuzzy-Evolutivo de otimização de asas de drone, com base em 2 laudas + artigo-base." | *A preencher pela equipe:* o que foi aceito, corrigido, rejeitado, testado e validado. Conferir que a árvore importa, que os endpoints respondem `placeholder:true`, e que nenhum algoritmo foi implementado no Portão A. |
| Claude Code (claude.ai/code) | **Implementação da lógica real (Portão B)**: baseline analítico (ponto crítico/Hessiana), sistema fuzzy Mamdani via scikit-fuzzy (membership/variables/rules/inference/system), PSO manual (`swarm`), busca operacional (`operate`), baselines (aleatória/gulosa), métricas/agregação, referência híbrida (`reference/target`), calibração real-coded adaptada de Kacimi et al. 2020 (`calibrate`), avaliador Mamdani vetorizado (`fast_eval`), runner de experimentos + persistência, visualizações (superfície/sensibilidade/convergência), e ligação de API/dashboard ao núcleo real. | "Implementar a lógica dos stubs na ordem dos marcos, sem alterar assinaturas, gerando números reais e mantendo os testes passando." | *A preencher pela equipe:* validar `pytest` (14 passam), conferir coerência (ótimo fuzzy ≈ analítico, MSE depois < antes, PSO ≥ baselines) e revisar criticamente o avaliador vetorizado vs scikit-fuzzy. |

## Itens NÃO gerados por IA / responsabilidade da equipe
- Decisões de modelagem (escolha das variáveis, termos linguísticos, justificativa das regras).
- Implementação e validação final dos algoritmos (fuzzy, PSO, calibração) — Portão B.
- Interpretação crítica dos resultados, limitações e conclusões.
- Conferência de que o código corresponde ao relatório e à apresentação.

## Observações
- O artigo-base e as referências são fontes reais (não inventadas pela IA).
- Nenhum número de resultado foi fabricado: **todos** os valores do relatório vêm de execução real
  persistida em `experiments/results/` (gerada por `scripts/run_experiments.py`).
- **Decisão técnica de desempenho (Portão B):** a calibração usa um avaliador Mamdani vetorizado em
  NumPy (`src/dfpo/fuzzy/fast_eval.py`) no laço interno do PSO, pois o scikit-fuzzy puro seria
  ~135× mais lento. O scikit-fuzzy permanece o sistema **oficial**; a equivalência numérica entre os
  dois é verificada automaticamente (`tests/test_pso.py::test_fast_evaluator_matches_scikit_fuzzy`,
  diferença < 0,5 na escala 0–100).
