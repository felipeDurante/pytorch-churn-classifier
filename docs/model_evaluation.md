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
| Dummy               | Preencher | Preencher | Preencher | Preencher  |
| Logistic Regression | Preencher | Preencher | Preencher | Preencher  |
| MLP                 | Preencher | Preencher | Preencher | Preencher  |

---

## Melhor Modelo

### Critério Principal

ROC-AUC

### Critério Secundário

Cost Saved

Descrever qual modelo apresentou melhor desempenho e justificar a escolha.

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

---

## Evidências

Inserir capturas de tela:

### Dashboard do MLflow

Imagem aqui

### Comparação de Métricas

Imagem aqui

### Curvas ROC

Imagem aqui

### Curvas Precision-Recall

Imagem aqui

---

## Conclusão

Descrever:

* Melhor modelo encontrado.
* Benefícios para o negócio.
* Limitações da solução.
* Próximos passos.
