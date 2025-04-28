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

1. **Extração de Dados**\
   Extração dos dados públicos de sessões de usuários do BigQuery (`google_analytics_sample`).

2. **Feature Engineering**

   - Transformações como cálculo de **tempo médio por página** e **ticket médio**.
   - Criação de variáveis derivadas de comportamento para inferência de churn.

3. **Balanceamento de Classes**\
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

| Métrica  | Valor      |
| -------- | ---------- |
| F1-Score | **0.9994** |
| Acurácia | **99.9%**  |
| Precisão | **99.9%**  |
| Recall   | **99.9%**  |

✅ **Modelo selecionado:** Random Forest

---

## ⚠️ Observação Crítica sobre a Acurácia Alta

Apesar dos resultados quase perfeitos, é importante destacar:

- O balanceamento via **SMOTE** pode ter gerado uma base artificialmente equilibrada.
- O modelo pode estar sofrendo de **overfitting**.
- A base utilizada é pública e genérica, sem dados reais de churn.

🔴 **Melhoria futura:**

- Testes com bases reais.
- Regularização e ajuste fino de hiperparâmetros.
- Validação Out-of-Sample.

---

## 🚀 Teste o Deploy Online

- **V1 API Deploy:** [https://churn-api-271029717147.southamerica-east1.run.app](https://churn-api-271029717147.southamerica-east1.run.app)
- **V2 API Deploy:** [https://churn-api-v2-271029717147.southamerica-east1.run.app](https://churn-api-v2-271029717147.southamerica-east1.run.app)

**Exemplo de chamada:**

```json
{
  "pageviews": 10,
  "timeOnSite": 300.5,
  "tempo_por_pagina": 30.05,
  "ticket_medio": 50000.0
}
```

Resposta esperada:

```json
{
  "prediction": 0,
  "message": "Cliente deve permanecer"
}
```

---

## 🖥️ Teste o Dashboard Localmente

```bash
# Clone o projeto
git clone https://github.com/rodrigo-rezend/projeto-churn-predict.git

# Instale as dependências
pip install -r requirements.txt

# Rode a API
uvicorn app:app --reload

# Rode o Dashboard de Previsão
streamlit run dashboard.py

# Rode o Dashboard Analítico
streamlit run dashboard_analytics.py
```

---

## 📦 Estrutura do Projeto

```
.
├── app.py
├── dashboard.py
├── dashboard_analytics.py
├── fetch_data.py
├── process_data.py
├── train_model.py
├── Dockerfile
├── requirements.txt
├── README.md
├── data/
├── figures/
└── models/
```

---

## 💬 Dificuldades Encontradas

- **Limitações de dados**: Base pública com limitações para predizer churn real.
- **Overfitting potencial**: Necessidade de avaliações futuras mais robustas.
- **Deploy no GCP**: Erros de permissão e ajustes finos de configuração.
- **Dashboard responsivo**: Adaptação do layout para melhor usabilidade.

Esses desafios agregaram muito à experiência técnica e de deploy prático!

---

## ✨ Autor

Desenvolvido por **Rodrigo Rezende** 🚀\
Contato: [https://www.linkedin.com/in/rodrigo-rezend/] ou [rodrigorezendemagalhaes@gmail.com]

---

```
```
