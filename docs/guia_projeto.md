# Guia do Projeto de Forecasting

## Objetivo
Prever vendas de produtos utilizando dados históricos, variáveis de concorrência e calendário, com modelos de machine learning.

## Estrutura de Pastas
- **app/**: Aplicações e scripts principais
- **data/**: Dados brutos e processados
    - **raw/**: Dados originais de vendas e concorrência
    - **processed/**: Dados tratados e prontos para modelagem
- **docs/**: Documentação do projeto
- **models/**: Modelos treinados salvos
- **notebooks/**: Notebooks de análise, treinamento e previsão

## Dados
- `ventas.csv`: Vendas históricas
- `competencia.csv`: Preços dos concorrentes
- `ventas_2025_inferencia.csv`: Dados para previsão futura

## Fluxo do Projeto
1. **Preparação dos dados**: Limpeza, integração e criação de variáveis
2. **Análise exploratória**: Visualizações e estatísticas
3. **Engenharia de variáveis**: Lags, médias móveis, calendário, preços
4. **Treinamento do modelo**: HistGradientBoostingRegressor
5. **Validação**: Métricas e gráficos comparativos
6. **Previsão**: Aplicação do modelo em dados futuros
7. **Documentação**: Orientações e explicações

## Como Executar
1. Instale as dependências do `requirements.txt`
2. Execute os notebooks em `notebooks/` na ordem:
   - `treinamento.ipynb` para preparar dados e treinar o modelo
   - `forecasting.ipynb` para gerar previsões

## Aplicação Web
Você pode testar o projeto online em: https://simula-venda.streamlit.app/

## Principais Bibliotecas
- pandas, numpy, matplotlib, seaborn, scikit-learn, holidays

## Observações
- Os arquivos processados são salvos em `data/processed/`
- O modelo final é salvo em `models/modelo_final.joblib`
- Consulte este guia para dúvidas sobre o fluxo ou estrutura
