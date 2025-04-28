"""
process_data.py
---------------
Script de processamento de dados brutos de sessões para criação do dataset de treinamento.

Objetivo:
- Realizar limpeza, tratamento de valores nulos e engenharia de atributos (feature engineering).
- Balancear o dataset para o problema de classificação binária de Churn usando SMOTE.

Principais Transformações:
- Criação de features derivadas: tempo médio por página, ticket médio.
- Identificação de padrões de comportamento como 'visitante rápido' e 'cliente de ticket alto'.

Impacto:
- Garante um conjunto de dados mais representativo e robusto para o treinamento dos modelos.
- Ajuda a melhorar a capacidade preditiva e generalização do modelo.

Resultado:
- Base processada salva em 'data/processed_sessions.csv' pronta para treinamento.
"""




import pandas as pd
import os
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def process_data():
    # Carregar o CSV
    df = pd.read_csv('data/ga_sessions_sample.csv', dtype={'fullVisitorId': str})
    print("Dados carregados! Shape inicial:", df.shape)

    # Tratar valores nulos
    df['pageviews'] = df['pageviews'].fillna(0).astype(int)
    df['timeOnSite'] = df['timeOnSite'].fillna(0).astype(float)
    df['transactions'] = df['transactions'].fillna(0).astype(int)
    df['transactionRevenue'] = df['transactionRevenue'].fillna(0).astype(float)

    # Criar a variável alvo (churn)
    # Churn = 1 para quem não comprou (transactions = 0)
    df['churn'] = df['transactions'].apply(lambda x: 1 if x == 0 else 0)

    # Feature Engineering
    df['tempo_por_pagina'] = df['timeOnSite'] / (df['pageviews'] + 1)
    df['ticket_medio'] = df['transactionRevenue'] / (df['transactions'] + 1)

    # Criar flags para engajamento
    df['engajamento_baixo'] = (df['pageviews'] <= 2).astype(int)
    df['visitante_rapido'] = (df['tempo_por_pagina'] <= 5).astype(int)
    df['cliente_ticket_alto'] = (df['ticket_medio'] > df['ticket_medio'].median()).astype(int)

    # Simplificar device (mobile, desktop, tablet)
    df['device_mobile'] = (df['device'] == 'mobile').astype(int)
    df['device_tablet'] = (df['device'] == 'tablet').astype(int)
    df['device_desktop'] = (df['device'] == 'desktop').astype(int)

    # Origem da visita
    df['via_organica'] = df['traffic_medium'].apply(lambda x: 1 if x == 'organic' else 0)
    df['via_pago'] = df['traffic_medium'].apply(lambda x: 1 if x == 'cpc' else 0)

    # Selecionar apenas as colunas que vamos usar
    features = [
        'pageviews',
        'timeOnSite',
        'tempo_por_pagina',
        'ticket_medio',
        'engajamento_baixo',
        'visitante_rapido',
        'cliente_ticket_alto',
        'device_mobile',
        'device_tablet',
        'device_desktop',
        'via_organica',
        'via_pago',
        'churn'
    ]
    df_final = df[features]
    print("Shape após feature engineering:", df_final.shape)

    # Separar X e y
    X = df_final.drop('churn', axis=1)
    y = df_final['churn']

    # Aplicar SMOTE apenas se necessário
    if len(y.unique()) > 1:
        print("Aplicando SMOTE para balanceamento...")
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        df_resampled = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.DataFrame(y_resampled, columns=['churn'])], axis=1)
        print("Balanceamento SMOTE realizado! Shape:", df_resampled.shape)
    else:
        print("SMOTE não aplicado. Apenas uma classe no target.")
        df_resampled = df_final.copy()

    # Salvar base processada
    os.makedirs('data', exist_ok=True)
    df_resampled.to_csv('data/processed_sessions.csv', index=False)
    print("Base processada salva em: data/processed_sessions.csv")

if __name__ == "__main__":
    process_data()
