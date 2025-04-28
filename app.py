"""
app.py
-------
API desenvolvida usando FastAPI para previsão de Churn de clientes.

Objetivo:
- Receber dados de sessões de clientes.
- Prever a probabilidade de um cliente realizar churn (abandono).
- Facilitar integrações com sistemas e plataformas empresariais.

Impacto:
- Permite que empresas identifiquem clientes em risco de abandono.
- Ajuda a direcionar estratégias de retenção e campanhas de marketing personalizadas.
"""



from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Inicializar o app
app = FastAPI()

# Carregar o modelo treinado
model = joblib.load('models/churn_model.pkl')

# Definir o esquema completo de entrada de dados
class CustomerData(BaseModel):
    pageviews: int
    timeOnSite: float
    tempo_por_pagina: float
    ticket_medio: float
    engajamento_baixo: int
    visitante_rapido: int
    cliente_ticket_alto: int
    device_mobile: int
    device_tablet: int
    device_desktop: int
    via_organica: int
    via_pago: int

# Rota principal
@app.get("/")
def read_root():
    return {"message": "API Churn v2.0 rodando!"}

# Rota de previsão
@app.post("/predict/")
def predict(data: CustomerData):
    # Transformar os dados recebidos em DataFrame
    input_data = pd.DataFrame([data.dict()])

    # Fazer a previsão
    prediction = model.predict(input_data)[0]

    # Interpretar o resultado
    result = "Cliente deve permanecer" if prediction == 0 else "Cliente com risco de churn"

    return {"prediction": int(prediction), "message": result}
