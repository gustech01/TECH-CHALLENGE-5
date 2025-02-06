import streamlit as st
import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

@st.cache_data(show_spinner=True)
def carregar_dados(caminho):
    """Carrega um dataset CSV, retorna o DataFrame ou erro."""
    try:
        return pd.read_csv(caminho)
    except FileNotFoundError:
        st.error(f"Arquivo não encontrado: {caminho}")
        return pd.DataFrame()

@st.cache_resource(show_spinner=True)
def carregar_imagem(caminho):
    """Carrega o caminho da imagem."""
    imagem_path = Path(caminho)
    if imagem_path.is_file():
        return str(imagem_path)
    else:
        st.error(f"Imagem não encontrada: {caminho}")
        return None

def show():
    # Logo FIAP
    left, cent, right = st.columns(3)
    with right:
        imagem = carregar_imagem('imagens/fiap.png')
        if imagem:
            st.image(imagem)

    # Título
    st.title('Modelos Preditivos')

    # Layout do aplicativo
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(['Matriz Multinomial', 'Curva ROC Multinomial', 'Matriz XGBoost', 'Curvas ROC XGBoost', 'Matriz Rede Neural', 'Curva Rede Neural'])

    # Carregando as matrizes de confusão e curvas ROC
    matriz_multinomial = carregar_dados("dataset/matriz_confusao_multinomial.csv")
    curva_roc_multinomial = carregar_dados("dataset/curvas_roc_multinomial.csv")
    
    matriz_xgboost = carregar_dados("dataset/matriz_confusao_xgboost.csv")
    curva_roc_xgboost = carregar_dados("dataset/curvas_roc_xgboost.csv")
    
    matriz_rede_neural = carregar_dados("dataset/matriz_confusao_rede_neural.csv")
    curva_roc_rede_neural = carregar_dados("dataset/curvas_roc_rede_neural.csv")

    if matriz_multinomial.empty or curva_roc_multinomial.empty or \
       matriz_xgboost.empty or curva_roc_xgboost.empty or \
       matriz_rede_neural.empty or curva_roc_rede_neural.empty:
        st.error("Os arquivos de matriz de confusão ou curvas ROC não foram carregados corretamente.")
        return

    with tab1:
        st.subheader("Matriz de Confusão - Regressão Multinomial")
        fig, ax = plt.subplots()
        sns.heatmap(matriz_multinomial, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title("Matriz de Confusão - Regressão Multinomial")
        st.pyplot(fig)

    with tab2:
        st.subheader("Curvas ROC - Regressão Multinomial")
        fig, ax = plt.subplots()
        for classe in curva_roc_multinomial['Classe'].unique():
            dados_classe = curva_roc_multinomial[curva_roc_multinomial['Classe'] == classe]
            ax.plot(dados_classe['FPR'], dados_classe['TPR'], label=f"{classe} (AUC = {auc(dados_classe['FPR'], dados_classe['TPR']):.2f})")
        ax.plot([0, 1], [0, 1], "k--")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Curvas ROC - Regressão Multinomial")
        ax.legend()
        st.pyplot(fig)

    with tab3:
        st.subheader("Matriz de Confusão - XGBoost")
        fig, ax = plt.subplots()
        sns.heatmap(matriz_xgboost, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title("Matriz de Confusão - XGBoost")
        st.pyplot(fig)

    with tab4:
        st.subheader("Curvas ROC - XGBoost")
        fig, ax = plt.subplots()
        for classe in curva_roc_xgboost['Classe'].unique():
            dados_classe = curva_roc_xgboost[curva_roc_xgboost['Classe'] == classe]
            ax.plot(dados_classe['FPR'], dados_classe['TPR'], label=f"{classe} (AUC = {auc(dados_classe['FPR'], dados_classe['TPR']):.2f})")
        ax.plot([0, 1], [0, 1], "k--")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Curvas ROC - XGBoost")
        ax.legend()
        st.pyplot(fig)

    with tab5:
        st.subheader("Matriz de Confusão - Rede Neural")
        fig, ax = plt.subplots()
        sns.heatmap(matriz_rede_neural, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title("Matriz de Confusão - Rede Neural")
        st.pyplot(fig)

    with tab6:
        st.subheader("Curvas ROC - Rede Neural")
        fig, ax = plt.subplots()
        for classe in curva_roc_rede_neural['Classe'].unique():
            dados_classe = curva_roc_rede_neural[curva_roc_rede_neural['Classe'] == classe]
            ax.plot(dados_classe['FPR'], dados_classe['TPR'], label=f"{classe} (AUC = {auc(dados_classe['FPR'], dados_classe['TPR']):.2f})")
        ax.plot([0, 1], [0, 1], "k--")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Curvas ROC - Rede Neural")
        ax.legend()
        st.pyplot(fig)

# Exibir o aplicativo
if __name__ == "__main__":
    show()
