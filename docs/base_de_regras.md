# Base de Regras — Fuzzy Mamdani (≥12 regras)

> ⚠️ **Portão A:** estrutura e justificativa das regras. A inferência ainda **não** está implementada
> (Portão B). As regras abaixo são o ponto de partida (16 regras efetivas, acima do mínimo de 12),
> cobrindo casos **típicos, intermediários, fronteiriços e conflitantes**, conforme exigido pelas
> laudas.

**Variáveis:** `x` envergadura (Curta/Média/Longa), `y` corda (Estreita/Média/Larga),
`z` carga útil (Leve/Padrão/Pesada) → `S` score de eficiência (Baixa/Regular/Excelente).
**Operadores:** conjunção = mínimo · agregação = máximo · defuzzificação = centróide.

> Princípio de domínio (asa de drone de entrega): existe uma geometria "ótima" intermediária
> (envergadura/corda moderadas) que maximiza eficiência; cargas maiores exigem mais área de asa
> (corda/envergadura maiores) para manter eficiência, mas geometrias exageradas penalizam por
> arrasto/peso estrutural. As regras abaixo codificam esse compromisso e devem ser **consistentes com
> o ótimo analítico** `(x≈2.35, y≈2.61)`.

| #  | x (envergadura) | y (corda) | z (carga) | → S (score) | Tipo / justificativa |
|----|-----------------|-----------|-----------|-------------|----------------------|
| R01 | Média  | Média    | Padrão | **Excelente** | Típico — geometria balanceada na carga nominal (próx. ao ótimo analítico) |
| R02 | Curta  | Estreita | Leve   | **Regular**   | Típico — asa pequena só serve p/ carga leve |
| R03 | Curta  | Estreita | Pesada | **Baixa**     | Crítico/conflitante — asa pequena não sustenta carga pesada |
| R04 | Longa  | Larga    | Pesada | **Regular**   | Típico — área grande sustenta carga, mas penaliza arrasto/peso |
| R05 | Longa  | Larga    | Leve   | **Baixa**     | Conflitante — asa superdimensionada p/ carga leve (ineficiente) |
| R06 | Média  | Média    | Leve   | **Regular**   | Intermediário — boa geometria, porém subutilizada |
| R07 | Média  | Média    | Pesada | **Regular**   | Intermediário — geometria boa no limite de carga |
| R08 | Média  | Estreita | Padrão | **Regular**   | Intermediário — corda insuficiente reduz sustentação |
| R09 | Média  | Larga    | Padrão | **Regular**   | Intermediário — corda excessiva adiciona arrasto |
| R10 | Curta  | Média    | Padrão | **Baixa**     | Fronteiriço — envergadura curta limita desempenho |
| R11 | Longa  | Média    | Padrão | **Regular**   | Fronteiriço — envergadura longa aceitável, leve perda |
| R12 | Curta  | Larga    | Padrão | **Baixa**     | Conflitante — proporção ruim (curta + larga) |
| R13 | Longa  | Estreita | Padrão | **Baixa**     | Conflitante — proporção ruim (longa + estreita) |
| R14 | Média  | Larga    | Pesada | **Excelente** | Típico — mais área de asa casa com carga pesada |
| R15 | Curta  | Estreita | Padrão | **Baixa**     | Fronteiriço — asa pequena no limite de carga nominal |
| R16 | Longa  | Larga    | Padrão | **Regular**   | Intermediário — área grande na carga nominal, leve excesso |

**Cobertura declarada:**
- *Típicos:* R01, R02, R04, R14
- *Intermediários:* R06, R07, R08, R09, R16
- *Fronteiriços:* R10, R11, R15
- *Conflitantes:* R05, R12, R13
- *Crítico:* R03

> Observação: no Portão B as regras serão materializadas como `skfuzzy.control.Rule`. Pesos de regra
> e/ou os parâmetros das MFs poderão ser ajustados pelo PSO (calibração); a tabela "depois" será
> registrada aqui ao lado desta versão inicial.
