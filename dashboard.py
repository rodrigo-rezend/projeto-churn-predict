"""
dashboard.py
-------------
Aplicativo Streamlit para upload de dados e visualização de análises de Churn.

Objetivo:
- Permitir que analistas e gestores explorem as previsões de churn de forma visual.
- Geração automática de gráficos e insights baseados nos dados enviados.

Impacto:
- Apoia a tomada de decisões estratégicas baseadas em dados.
- Facilita a compreensão do comportamento dos clientes e a identificação de padrões de churn.
"""


import streamlit as st
import pandas as pd
import joblib

# Carregar modelo treinado
model = joblib.load('models/churn_model.pkl')

# Título
st.title('🔮 Dashboard de Previsão de Churn')

# Instruções
st.markdown("""
### Como utilizar:
1. Baixe o arquivo de exemplo clicando no botão abaixo.
2. Preencha ou edite o arquivo mantendo exatamente as mesmas colunas.
3. Faça upload do arquivo CSV preenchido.
4. Clique no botão "Prever Churn" para gerar as previsões.
""")

# Gerar exemplo de CSV
def gerar_exemplo_csv():
    data = {
        "pageviews": [8, 5, 10, 3, 12],
        "timeOnSite": [350.0, 120.0, 600.0, 90.0, 720.0],
        "tempo_por_pagina": [43.75, 24.0, 60.0, 30.0, 60.0],
        "ticket_medio": [1200.0, 800.0, 2000.0, 500.0, 1800.0],
        "engajamento_baixo": [0, 1, 0, 1, 0],
        "visitante_rapido": [0, 1, 0, 1, 0],
        "cliente_ticket_alto": [1, 0, 1, 0, 1],
        "device_mobile": [1, 1, 0, 1, 0],
        "device_tablet": [0, 0, 0, 0, 1],
        "device_desktop": [0, 0, 1, 0, 0],
        "via_organica": [1, 0, 1, 0, 1],
        "via_pago": [0, 1, 0, 1, 0]
    }
    df_exemplo = pd.DataFrame(data)
    return df_exemplo

# Botão para baixar o exemplo
exemplo_csv = gerar_exemplo_csv()
csv_exemplo = exemplo_csv.to_csv(index=False).encode('utf-8')

st.download_button(
    label="📥 Baixar Exemplo de CSV",
    data=csv_exemplo,
    file_name='exemplo_upload_churn.csv',
    mime='text/csv'
)

st.divider()

# Upload do arquivo CSV
uploaded_file = st.file_uploader("📂 Faça upload do seu arquivo CSV:", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader('👀 Pré-visualização dos dados:')
    st.dataframe(df.head())

    if st.button('🚀 Prever Churn'):
        predictions = model.predict(df)
        df['churn_prediction'] = predictions

        st.subheader('📊 Resultados da Previsão:')
        st.dataframe(df[['churn_prediction']].value_counts().rename('Quantidade').reset_index())

        st.subheader('📋 Dados com Previsão:')
        st.dataframe(df)

        csv_resultado = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Baixar resultados em CSV",
            data=csv_resultado,
            file_name='churn_predictions.csv',
            mime='text/csv'
        )
