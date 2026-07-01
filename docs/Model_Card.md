# Model Card — Telco Churn Classifier

**Nome do modelo:** Telco Churn Classifier

**Versão:** 0.1.0

## Objetivo
Prever a probabilidade de churn de clientes de telecomunicações para apoiar ações de retenção.

## Uso pretendido
- Priorização de campanhas de retenção.
- Apoio à análise de risco de churn por cliente.

## Uso não indicado
- Decisões totalmente automatizadas sem revisão humana.
- Cenários com mudança de domínio sem revalidação do modelo.

## Dados
- Fonte: Telco Customer Churn.
- Variável-alvo: `Churn Value`.
- Pré-processamento: remoção de colunas de vazamento, imputação, one-hot encoding e padronização.
- Features principais utilizadas no pipeline: variáveis cadastrais, contratuais, de serviço e de cobrança presentes na base original.

## Performance
- Métricas avaliadas: Accuracy, Precision, Recall, F1-Score, ROC-AUC, PR-AUC e Cost Saved.
- Comparação contra baselines: Dummy Classifier e Logistic Regression.
- Resultados consolidados: ver [docs/model_evaluation.md](docs/model_evaluation.md).
- Resultado consolidado atual: o MLP apresentou o melhor ROC-AUC médio, o melhor PR-AUC médio e o maior Cost Saved médio.

## Limitações
- Dataset desbalanceado.
- Resultado sensível ao limiar de decisão.
- Features geográficas e socioeconômicas podem introduzir ruído e viés.

## Vieses conhecidos
- Potencial viés por região, tipo de contrato e perfil de consumo.
- Não foram validados subgrupos sensíveis de forma separada.

## Cenários de falha
- Mudança de distribuição dos dados em produção.
- Clientes fora do padrão histórico do dataset.
- Payload incompleto ou fora do esquema esperado pela API.

## Reprodutibilidade
- Seed centralizada em `src/projeto_ml/config.py`.
- Validação cruzada estratificada com `shuffle=True` e `random_state` fixo.
- Artefatos de comparação e rastreio registrados no MLflow local.

## Artefatos
- Pipeline serializado em `models/baseline_pipeline.joblib`.
- Runs e artefatos registrados no MLflow.

## Observações
- Esta model card deve ser atualizada após novos treinamentos e alterações relevantes no conjunto de features ou no esquema de entrada.
