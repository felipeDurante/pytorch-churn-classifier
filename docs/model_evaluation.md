# Avaliação dos Modelos

## Objetivo

Comparar os modelos avaliados para predição de churn e identificar a melhor alternativa para utilização em produção.

---

## Modelos Avaliados

### Dummy Classifier

Baseline utilizado como referência mínima de desempenho.

### Logistic Regression

Modelo linear utilizado como baseline principal.

### Multi Layer Perceptron (PyTorch)

Rede neural feedforward treinada utilizando PyTorch com Early Stopping e validação cruzada.

---

## Estratégia Experimental

### Validação

* Stratified K-Fold
* 5 Folds
* Seed fixa

### Métricas

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC
* PR-AUC
* Cost Saved

---

## Resultados

| Modelo              | F1        | ROC-AUC   | PR-AUC    | Cost Saved |
| ------------------- | --------- | --------- | --------- | ---------- |
| Dummy               | 0.0000    | 0.5000    | 0.2654    | 0.00       |
| Logistic Regression | 0.6051    | 0.8474    | 0.6485    | 21954.00   |
| MLP                 | 0.6230    | 0.8525    | 0.6623    | 23370.00   |

---

## Melhor Modelo

### Critério Principal

ROC-AUC

### Critério Secundário

Cost Saved

O MLP foi o melhor modelo consolidado, com maior ROC-AUC, maior PR-AUC e maior Cost Saved médio. A Logistic Regression segue como baseline forte e mais simples, mas ficou abaixo do MLP nos critérios principais.

---

## Trade-off de Negócio

### Falso Positivo

Cliente identificado como churn quando não iria cancelar.

Impacto:

* Custo de retenção desnecessário.

### Falso Negativo

Cliente que cancela sem ser identificado pelo modelo.

Impacto:

* Perda de receita.
* Perda do cliente.

---

## MLflow

Todos os experimentos foram registrados utilizando MLflow.

Artefatos registrados:

* Modelos treinados
* Métricas
* Parâmetros
* Gráficos de comparação

Ver também [docs/experiments.md](docs/experiments.md) para a documentação operacional do tracking.

---

## Evidências

Inserir capturas de tela:

### Dashboard do MLflow

Inserir captura do dashboard apontando runs, métricas e artefatos registrados.

### Comparação de Métricas

Inserir gráfico comparando os três modelos avaliados.

### Curvas ROC

Inserir curvas ROC dos modelos avaliados.

### Curvas Precision-Recall

Inserir curvas Precision-Recall dos modelos avaliados.

---

## Conclusão

Descrever:

* Melhor modelo encontrado: MLP.
* Benefícios para o negócio: maior capacidade de capturar churn com melhor equilíbrio entre recall e precisão.
* Limitações da solução: dependência da qualidade do dataset histórico e sensibilidade a drift.
* Próximos passos: validar em dados mais recentes, revisar threshold de decisão e publicar a API em ambiente controlado.
