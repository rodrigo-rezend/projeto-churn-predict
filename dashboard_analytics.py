"""
dashboard_analytics.py
------------------------
Dashboard desenvolvido em Streamlit para análise gráfica das previsões de Churn.

Objetivo:
- Fornecer uma visualização amigável dos dados processados.
- Gerar insights visuais através de gráficos de dispersão, histogramas e métricas de churn.
- Facilitar a interpretação do comportamento dos usuários que geram ou não churn.

Funcionalidades:
- Upload de arquivo CSV com colunas específicas para análise (pageviews, timeOnSite, tempo_por_pagina, ticket_medio).
- Geração de previsões de churn usando o modelo pré-treinado.
- Exibição de métricas de retenção vs churn.
- Visualizações gráficas automáticas para facilitar o entendimento dos padrões de comportamento dos usuários.

Impacto:
- Auxilia analistas e gestores a identificar rapidamente padrões de churn.
- Suporte na criação de campanhas de retenção personalizadas baseadas em dados históricos e predições.
- Reduz o custo de churn aumentando a eficácia das ações de retenção.

Nota:
- Este arquivo complementa o `dashboard.py` focando em visualizações analíticas.
- Pode ser utilizado tanto em ambiente local quanto ser futuramente dockerizado para demonstração online.
"""



import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Configurações iniciais
st.set_page_config(page_title="Dashboard Churn", layout="wide")
sns.set_style("whitegrid")

# Título principal
st.title('🔮 Dashboard de Previsão e Análise de Churn')

# Carregar modelo
model = joblib.load('models/churn_model.pkl')

# Instruções de uso
st.markdown("""
### 📋 Como utilizar:
1. Baixe o exemplo de CSV no botão abaixo.
2. Preencha ou edite mantendo as mesmas colunas.
3. Faça upload do seu arquivo CSV.
4. Clique no botão para prever churn e analisar!
""")

# Exemplo de CSV
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
    return pd.DataFrame(data)

exemplo_csv = gerar_exemplo_csv()
st.download_button(
    label="📥 Baixar Exemplo de CSV",
    data=exemplo_csv.to_csv(index=False).encode('utf-8'),
    file_name='exemplo_upload_churn.csv',
    mime='text/csv'
)

st.divider()

# Upload do arquivo
uploaded_file = st.file_uploader("📂 Faça upload do seu arquivo CSV:", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader('👀 Pré-visualização dos Dados:')
    st.dataframe(df.head())

    if st.button('🚀 Prever Churn e Analisar'):
        predictions = model.predict(df)
        df['churn_prediction'] = predictions

        st.success('✅ Previsões realizadas com sucesso! Veja abaixo os resultados.')

        st.divider()

        # Distribuição Churn
        st.header('📊 Distribuição de Churn')

        col1, col2 = st.columns([2, 1])

        with col1:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.countplot(x='churn_prediction', data=df, palette='pastel', ax=ax)
            ax.set_xlabel('Churn Prediction (0 = Permanece, 1 = Churn)', fontsize=12)
            ax.set_ylabel('Contagem', fontsize=12)
            ax.set_title('Distribuição de Clientes - Churn vs Permanência', fontsize=16)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)

        with col2:
            churn_percent = (df['churn_prediction'].sum() / df.shape[0]) * 100
            stay_percent = 100 - churn_percent

            fig_pie, ax_pie = plt.subplots(figsize=(4, 4))
            ax_pie.pie([stay_percent, churn_percent],
                       labels=['Permanecer', 'Churn'],
                       autopct='%1.1f%%',
                       colors=['#90ee90', '#ff9999'],
                       startangle=90,
                       explode=(0, 0.1))
            ax_pie.axis('equal')
            st.pyplot(fig_pie, use_container_width=True)

        st.divider()

        # Insights principais
        st.header('🧠 Insights Gerais')
        st.markdown(f"""
        - **{churn_percent:.2f}%** dos clientes têm alta probabilidade de churn.
        - **{stay_percent:.2f}%** dos clientes devem permanecer.
        """, unsafe_allow_html=True)

        st.divider()

        # Análises Comportamentais
        st.header('📈 Análises Comportamentais')

        col3, col4 = st.columns(2)

        with col3:
            st.subheader('Tempo no site (segundos)')
            fig1, ax1 = plt.subplots(figsize=(5, 3.5))
            sns.boxplot(x='churn_prediction', y='timeOnSite', data=df, palette='Set2', ax=ax1)
            ax1.set_xlabel('Churn Prediction')
            ax1.set_ylabel('Tempo no Site')
            fig1.tight_layout()
            st.pyplot(fig1, use_container_width=True)

        with col4:
            st.subheader('Número de Pageviews')
            fig2, ax2 = plt.subplots(figsize=(5, 3.5))
            sns.boxplot(x='churn_prediction', y='pageviews', data=df, palette='Set2', ax=ax2)
            ax2.set_xlabel('Churn Prediction')
            ax2.set_ylabel('Pageviews')
            fig2.tight_layout()
            st.pyplot(fig2, use_container_width=True)

        st.subheader('Ticket Médio por Grupo')
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        sns.boxplot(x='churn_prediction', y='ticket_medio', data=df, palette='Set3', ax=ax3)
        ax3.set_xlabel('Churn Prediction')
        ax3.set_ylabel('Ticket Médio (R$)')
        fig3.tight_layout()
        st.pyplot(fig3, use_container_width=True)

        churn_group = df.groupby('churn_prediction').agg({
            'timeOnSite': 'mean',
            'pageviews': 'mean',
            'ticket_medio': 'mean'
        }).reset_index()

        st.markdown(f"""
        ### 📋 Análises Detalhadas:
        - Permanência: tempo médio no site de **{churn_group.loc[churn_group['churn_prediction'] == 0, 'timeOnSite'].values[0]:.2f} segundos**.
        - Churn: tempo médio no site de **{churn_group.loc[churn_group['churn_prediction'] == 1, 'timeOnSite'].values[0]:.2f} segundos**.
        - Permanência: média de **{churn_group.loc[churn_group['churn_prediction'] == 0, 'pageviews'].values[0]:.2f} pageviews**.
        - Churn: média de **{churn_group.loc[churn_group['churn_prediction'] == 1, 'pageviews'].values[0]:.2f} pageviews**.
        - Permanência: ticket médio de **R$ {churn_group.loc[churn_group['churn_prediction'] == 0, 'ticket_medio'].values[0]:.2f}**.
        - Churn: ticket médio de **R$ {churn_group.loc[churn_group['churn_prediction'] == 1, 'ticket_medio'].values[0]:.2f}**.
        """, unsafe_allow_html=True)

        st.divider()

        # Baixar arquivo
        csv_resultado = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Baixar Dados com Previsão e Análise",
            data=csv_resultado,
            file_name='churn_predictions_analise.csv',
            mime='text/csv'
        )
