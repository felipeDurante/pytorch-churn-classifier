# Model Card — Telco Churn Classifier

**Nome do modelo:** Telco Churn Classifier (baseline)

**Versão:** 0.1.0

## Sumário
Breve descrição do objetivo do modelo: prever se um cliente irá cancelar (churn) baseado em informações demográficas e de serviço.

## Uso pretendido
- Diagnóstico e priorização de ações de retenção em ambiente controlado.
- Suporte a análises exploratórias e tomada de decisão interna.

## Uso não indicado
- Decisões automatizadas que impactem crédito, emprego ou elegibilidade sem revisão humana.
- Aplicações sem verificação de vieses e explicabilidade.

## Dados
- Fonte: Telco Customer Churn (arquivo original em `data/raw/`)
- Tamanho: 7.043 amostras (no dataset usado inicialmente)
- Colunas principais: `CustomerID`, `Tenure Months`, `Monthly Charges`, `Total Charges`, `Contract`, ...
- Pré-processamento: imputação de `Total Charges`, exclusão de colunas de vazamento (`Churn Label`, `Churn Score`, `Churn Reason`), codificação de variáveis categóricas e padronização de numéricas.

## Métricas de avaliação
- Métricas recomendadas: Recall, F1-score, ROC-AUC e PR-AUC.
- Baseline reportado: valores médios por validação cruzada estratificada (registrados no MLflow quando treinado).

## Reprodutibilidade
- Seed centralizada: `SEED = 42` (arquivo `src/projeto_ml/config.py`).
- Validação: `StratifiedKFold` com `shuffle=True` e `random_state=SEED`.
- Comando de treino: `python -m projeto_ml.train --data data/raw/Telco_customer_churn(Telco_Churn).csv`

## Limitações e vieses
- Dataset agrupado por alvo: é necessário embaralhar antes de dividir — caso não seja feito pode haver vazamento.
- Desbalanceamento: classe positiva (~26.5%) — métricas de acurácia são enganosas.
- Possíveis vieses socioeconômicos ou geográficos não investigados (colunas de localização foram excluídas inicialmente).

## Responsabilidade e mitigação
- Recomenda-se auditar desempenho por subgrupos sensíveis (idade, região, etc.).
- Documentar e revisar features de alta cardinalidade (city, zip) antes de inclusão em produção.

## Artefatos
- Modelo serializado: `models/baseline_pipeline.joblib` após treino
- Runs MLflow: `mlruns/` local (quando usado)

## Contato
- Mantido por: equipe do Tech Challenge
- Repositório: README.md no repositório principal

---

*Template gerado automaticamente — preencher métricas e detalhes pós-treino.*
