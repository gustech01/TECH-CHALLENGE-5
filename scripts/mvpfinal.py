import streamlit as st
import pandas as pd
from pathlib import Path
from sklearn.metrics import auc

@st.cache_data(show_spinner=True)
def carregar_dados(caminho):
    """Carrega um dataset CSV, retorna o DataFrame ou erro."""
    try:
        return pd.read_csv(caminho, usecols=lambda column: column not in ["Unnamed: 0"])
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
    matriz_multinomial = carregar_dados("datasets/matriz_confusao_multinomial.csv")
    curva_roc_multinomial = carregar_dados("datasets/curvas_roc_multinomial.csv")
    
    matriz_xgboost = carregar_dados("datasets/matriz_confusao_xgboost.csv")
    curva_roc_xgboost = carregar_dados("datasets/curvas_roc_xgboost.csv")
    
    matriz_rede_neural = carregar_dados("datasets/matriz_confusao_rede_neural.csv")
    curva_roc_rede_neural = carregar_dados("datasets/curvas_roc_rede_neural.csv")

    if matriz_multinomial.empty or curva_roc_multinomial.empty or \
       matriz_xgboost.empty or curva_roc_xgboost.empty or \
       matriz_rede_neural.empty or curva_roc_rede_neural.empty:
        st.error("Os arquivos de matriz de confusão ou curvas ROC não foram carregados corretamente.")
        return

    with tab1:
        st.subheader("Matriz de Confusão - Regressão Multinomial")
        st.write(matriz_multinomial)

    with tab2:
        st.subheader("Curvas ROC - Regressão Multinomial")
        for classe in curva_roc_multinomial['Classe'].unique():
            dados_classe = curva_roc_multinomial[curva_roc_multinomial['Classe'] == classe]
            st.line_chart(dados_classe.set_index('FPR')['TPR'], height=400, width=700)
            st.write(f"AUC - {classe}: {auc(dados_classe['FPR'], dados_classe['TPR']):.2f}")
        st.write("Linha de Referência")
        st.line_chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]}), height=400, width=700)

    with tab3:
        st.subheader("Matriz de Confusão - XGBoost")
        st.write(matriz_xgboost)

    with tab4:
        st.subheader("Curvas ROC - XGBoost")
        for classe in curva_roc_xgboost['Classe'].unique():
            dados_classe = curva_roc_xgboost[curva_roc_xgboost['Classe'] == classe]
            st.line_chart(dados_classe.set_index('FPR')['TPR'], height=400, width=700)
            st.write(f"AUC - {classe}: {auc(dados_classe['FPR'], dados_classe['TPR']):.2f}")
        st.write("Linha de Referência")
        st.line_chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]}), height=400, width=700)

    with tab5:
        st.subheader("Matriz de Confusão - Rede Neural")
        st.write(matriz_rede_neural)

    with tab6:
        st.subheader("Curvas ROC - Rede Neural")
        for classe in curva_roc_rede_neural['Classe'].unique():
            dados_classe = curva_roc_rede_neural[curva_roc_rede_neural['Classe'] == classe]
            st.line_chart(dados_classe.set_index('FPR')['TPR'], height=400, width=700)
            st.write(f"AUC - {classe}: {auc(dados_classe['FPR'], dados_classe['TPR']):.2f}")
        st.write("Linha de Referência")
        st.line_chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]}), height=400, width=700)

# Exibir o aplicativo
if __name__ == "__main__":
    show()
