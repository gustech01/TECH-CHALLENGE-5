import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Função para carregar dados
@st.cache_data(show_spinner=True)
def carregar_dados(caminho):
    """Carrega um dataset CSV, retorna o DataFrame ou erro."""
    try:
        return pd.read_csv(caminho)
    except FileNotFoundError:
        st.error(f"Arquivo não encontrado: {caminho}")
        return pd.DataFrame()

# Função para carregar imagens
@st.cache_resource(show_spinner=True)
def carregar_imagem(caminho):
    """Carrega o caminho da imagem."""
    imagem_path = Path(caminho)
    if imagem_path.is_file():
        return str(imagem_path)
    else:
        st.error(f"Imagem não encontrada: {caminho}")
        return None

# Função para exibir a matriz de confusão como gráfico
def plot_matriz_confusao(df, titulo):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(df, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(titulo)
    ax.set_xlabel("Previsões")
    ax.set_ylabel("Valores Reais")
    st.pyplot(fig)

# Interface principal
def show():
    # Adiciona imagem no topo (Logo FIAP)
    left, cent, right = st.columns(3)
    with right:
        imagem = carregar_imagem('imagens/fiap.png')
        if imagem:
            st.image(imagem)

    # Título
    st.title('Matrizes e Modelos')

    # Criando abas (tabs)
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        'Matriz - Multinomial', 
        'Curva ROC - Multinomial', 
        'Matriz - XGBoost', 
        'Curva ROC - XGBoost', 
        'Matriz - Rede Neural', 
        'Curva ROC - Rede Neural'
    ])

    # 🔹 Criando tabelas e gráficos dentro das abas
    with tab1:
        st.subheader("Matriz de Confusão - Multinomial")
        df_multinomial = carregar_dados('dados/matriz_multinomial.csv')
        if not df_multinomial.empty:
            st.dataframe(df_multinomial, height=200, width=400)
            plot_matriz_confusao(df_multinomial, "Matriz de Confusão - Multinomial")

    with tab2:
        st.subheader("Curva ROC - Multinomial")
        imagem_roc_multinomial = carregar_imagem('imagens/roc_multinomial.png')
        if imagem_roc_multinomial:
            st.image(imagem_roc_multinomial)

    with tab3:
        st.subheader("Matriz de Confusão - XGBoost")
        df_xgb = carregar_dados('dados/matriz_xgb.csv')
        if not df_xgb.empty:
            st.dataframe(df_xgb, height=200, width=400)
            plot_matriz_confusao(df_xgb, "Matriz de Confusão - XGBoost")

    with tab4:
        st.subheader("Curva ROC - XGBoost")
        imagem_roc_xgb = carregar_imagem('imagens/roc_xgb.png')
        if imagem_roc_xgb:
            st.image(imagem_roc_xgb)

    with tab5:
        st.subheader("Matriz de Confusão - Rede Neural")
        df_nn = carregar_dados('dados/matriz_nn.csv')
        if not df_nn.empty:
            st.dataframe(df_nn, height=200, width=400)
            plot_matriz_confusao(df_nn, "Matriz de Confusão - Rede Neural")

    with tab6:
        st.subheader("Curva ROC - Rede Neural")
        imagem_roc_nn = carregar_imagem('imagens/roc_nn.png')
        if imagem_roc_nn:
            st.image(imagem_roc_nn)

# Executar a interface no Streamlit
if __name__ == "__main__":
    show()
