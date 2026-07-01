# Arquitetura de Deploy

## Escolha: Real-time

A solução foi desenhada para **inferência em tempo real** via FastAPI.

## Justificativa
- A API responde sob demanda por cliente/registro.
- O tempo de resposta é importante para integração com front-end, CRM ou ferramentas operacionais.
- O modelo é pequeno o suficiente para ser carregado em memória e executado localmente.
- A arquitetura já possui serialização do pipeline e endpoint `/predict`.

## Quando batch faria sentido
- Processamento periódico de grandes volumes.
- Geração diária de scores para toda a base.
- Cenários sem necessidade de resposta imediata.

## Fluxo de deploy
1. Treinar o modelo.
2. Salvar o pipeline em `models/baseline_pipeline.joblib`.
3. Subir a API FastAPI.
4. Consumir `/predict` em tempo real.

## Observação
- O repositório também mantém o fluxo de MLflow local para rastreabilidade e comparação de experimentos.