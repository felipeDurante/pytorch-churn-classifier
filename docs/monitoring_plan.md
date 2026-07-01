# Plano de Monitoramento

## Objetivo
Monitorar qualidade, disponibilidade e saúde do modelo em produção.

## Métricas
- Latência p95 e p99 da API.
- Taxa de erro HTTP 4xx/5xx.
- Volume de requisições por intervalo.
- Distribuição das features de entrada.
- Drift de dados e de predições.
- Taxa de churn previsto vs. realizado.

## Alertas
- Latência acima do limite acordado.
- Aumento de erro 5xx.
- Drift estatístico persistente.
- Queda de performance em amostras rotuladas.

## Playbook de resposta
1. Verificar saúde da API e logs.
2. Confirmar integridade do modelo e do payload.
3. Analisar drift e dados recentes.
4. Reexecutar treinamento se houver degradação sustentada.
5. Promover rollback se o modelo novo degradar o serviço.

## Instrumentação sugerida
- Logs estruturados.
- Métrica de latência por requisição.
- Monitoramento de drift e qualidade com checkpoints periódicos.
- Registro de métricas no MLflow para comparação entre versões.