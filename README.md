# Previsão de Churn - Machine Learning, FastAPI, Streamlit e Deploy no Google Cloud ☁️

Este projeto visa construir uma solução de ponta a ponta para a **previsão de churn de clientes** utilizando Machine Learning, APIs modernas e dashboards analíticos. Foi desenvolvido como parte de um portfólio focado em Ciência de Dados aplicada à geração de insights de negócio.

---

## 🔥 Tecnologias Utilizadas
- **Python 3.11**
- **Pandas** - Manipulação de dados
- **Scikit-Learn** - Modelos tradicionais de Machine Learning
- **XGBoost** e **LightGBM** - Modelos de Gradient Boosting
- **SMOTE** - Balanceamento de dados
- **FastAPI** - API de inferência de modelos
- **Streamlit** - Dashboard interativo
- **Docker** - Conteinerização
- **Google Cloud Run** - Deploy serverless

---

## 🛠️ Pipeline de Construção do Projeto

1. **Extração de Dados**  
   Extração dos dados públicos de sessões de usuários do BigQuery (`google_analytics_sample`).

2. **Feature Engineering**  
   - Transformações como cálculo de **tempo médio por página** e **ticket médio**.
   - Criação de variáveis derivadas de comportamento para inferência de churn.

3. **Balanceamento de Classes**  
   Utilização do **SMOTE** para equilibrar as classes minoritárias e reduzir o viés do modelo.

4. **Treinamento e Avaliação de Modelos**  
   - Modelos treinados: **Random Forest**, **XGBoost** e **LightGBM**.
   - Avaliação via **Validação Cruzada (5 folds)**.
   - Seleção do melhor modelo baseado no **F1-Score**.

5. **Deploy do Modelo**  
   - Criação de uma API REST com FastAPI.
   - Conteinerização com Docker.
   - Deploy serverless via Google Cloud Run.

6. **Criação de Dashboards**  
   - Dashboard de Upload e Previsão de novos dados.
   - Dashboard Analítico para geração de insights visuais.

---

## 📈 Resultados do Modelo

| Métrica | Valor |
|:--------|:------|
| F1-Score | **0.9994** |
| Acurácia | **99.9%** |
| Precisão | **99.9%** |
| Recall | **99.9%** |

✅ **Modelo selecionado:** Random Forest

---

## ⚠️ Observação Crítica sobre a Acurácia Alta

Apesar dos resultados quase perfeitos nas métricas de avaliação, **é importante destacar**:

- O balanceamento via **SMOTE** pode ter gerado uma base artificialmente equilibrada.
- O modelo pode estar sofrendo de **overfitting** (aprendizado excessivo dos padrões gerados pelo SMOTE).
- Os dados disponíveis (Google Analytics Sample) são de navegação genérica e podem não refletir perfeitamente um cenário real de churn de uma empresa.

🔴 **Melhoria futura:**  
Para validar melhor a robustez do modelo, seria ideal:

- Utilizar uma base de dados real de clientes (não apenas navegação).
- Aplicar técnicas de regularização, tuning de hiperparâmetros mais avançado.
- Avaliar em um conjunto de testes externo (out-of-sample) sem balanceamento.

---

## 🚀 Teste o Deploy Online

- **V1 API Deploy:** [https://churn-api-271029717147.southamerica-east1.run.app](https://churn-api-271029717147.southamerica-east1.run.app/docs)
- **V2 API Deploy:** [https://churn-api-v2-271029717147.southamerica-east1.run.app](https://churn-api-v2-271029717147.southamerica-east1.run.app/docs)

### Teste com um JSON no endpoint `/predict/`:

```json
{
  "pageviews": 10,
  "timeOnSite": 300.5,
  "tempo_por_pagina": 30.05,
  "ticket_medio": 50000.0
}
