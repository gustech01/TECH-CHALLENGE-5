import streamlit as st
import pandas as pd

# Função para carregar dados
@st.cache_data(show_spinner=True)
def carregar_dados(caminho):
    try:
        return pd.read_csv(caminho)
    except FileNotFoundError:
        st.error(f"Arquivo não encontrado: {caminho}")
        return pd.DataFrame()

# Interface principal
def show():
    st.title('Matrizes de Confusão e Curvas ROC')

    # Criando abas (tabs)
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        'Matriz - Multinomial', 
        'Curva ROC - Multinomial', 
        'Matriz - XGBoost', 
        'Curva ROC - XGBoost', 
        'Matriz - Rede Neural', 
        'Curva ROC - Rede Neural'
    ])

    # 🔹 Matrizes de Confusão (Heatmaps)
    def exibir_matriz(df, modelo):
        if df.empty:
            st.error(f"Dados da matriz ({modelo}) não encontrados!")
            return
        st.subheader(f"Matriz de Confusão - {modelo}")
        st.write(df.style.background_gradient(cmap="Blues"))  # Heatmap no Streamlit

    with tab1:
        df_multinomial = carregar_dados('dados/matriz_multinomial.csv')
        exibir_matriz(df_multinomial, "Multinomial")

    with tab3:
        df_xgb = carregar_dados('dados/matriz_xgb.csv')
        exibir_matriz(df_xgb, "XGBoost")

    with tab5:
        df_nn = carregar_dados('dados/matriz_nn.csv')
        exibir_matriz(df_nn, "Rede Neural")

    # 🔹 Curvas ROC (Line Chart)
    def exibir_curva_roc(df, modelo):
        if df.empty:
            st.error(f"Dados da Curva ROC ({modelo}) não encontrados!")
            return
        st.subheader(f"Curva ROC - {modelo}")
        st.line_chart(df)  # Plota automaticamente as colunas

    with tab2:
        df_roc_multinomial = carregar_dados('dados/roc_multinomial.csv')
        exibir_curva_roc(df_roc_multinomial, "Multinomial")

    with tab4:
        df_roc_xgb = carregar_dados('dados/roc_xgb.csv')
        exibir_curva_roc(df_roc_xgb, "XGBoost")

    with tab6:
        df_roc_nn = carregar_dados('dados/roc_nn.csv')
        exibir_curva_roc(df_roc_nn, "Rede Neural")

# Executar a interface no Streamlit
if __name__ == "__main__":
    show()
