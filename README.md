# 📈 Projeto de Forecasting de Vendas

Este projeto tem como objetivo construir um pipeline completo de previsão de vendas utilizando dados históricos, engenharia de variáveis, validação de modelos e geração de inferências para o ano de 2025.

## Estrutura do Projeto
- **app/**: Código principal do aplicativo (em desenvolvimento)
- **data/**: Dados utilizados no projeto
  - **raw/**: Dados brutos de treinamento e inferência
  - **processed/**: Dados processados e prontos para modelagem
- **docs/**: Documentação complementar (em desenvolvimento)
- **models/**: Modelos treinados e arquivos de exportação
- **notebooks/**: Notebooks de análise, treinamento e inferência

## Principais Etapas
1. **Carregamento dos dados**: Importação dos arquivos de vendas e concorrentes.
2. **Validação e limpeza**: Análise de qualidade dos dados, tratamento de nulos e tipos.
3. **Engenharia de variáveis**: Criação de variáveis temporais, lags, médias móveis, descontos, preços de concorrentes e codificação one-hot.
4. **Modelagem**: Treinamento do modelo HistGradientBoostingRegressor com validação e comparação com baseline.
5. **Inferência**: Preparação dos dados de 2025, aplicação do modelo final e geração de previsões.
6. **Exportação**: Salvamento dos resultados e do modelo final para uso futuro.

## Como Executar
1. Instale as dependências necessárias:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn holidays joblib
   ```
2. Execute os notebooks na pasta `notebooks/` para seguir o fluxo de análise, treinamento e inferência.
3. O modelo final será salvo em `models/modelo_final.joblib` e os dados processados em `data/processed/`.

## Requisitos
- Python 3.8+
- Bibliotecas: pandas, numpy, matplotlib, seaborn, scikit-learn, holidays, joblib

## Autor
Projeto desenvolvido por Gabriel Marcano para o desafio DS4B.

---
Dúvidas ou sugestões? Entre em contato!
