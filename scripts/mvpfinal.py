import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path
from PIL import Image

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
        return Image.open(imagem_path)
    else:
        st.error(f"Imagem não encontrada: {caminho}")
        return None

def plot_curvas_roc_altair(curva_roc, titulo):
    """
    Plota as curvas ROC usando a biblioteca Altair.
    """
    curva_roc['AUC_label'] = curva_roc['Classe'] + " (AUC = " + curva_roc['AUC'].round(2).astype(str) + ")"

    # Criar gráfico com Altair
    chart = alt.Chart(curva_roc).mark_line().encode(
        x=alt.X('FPR:Q', title='False Positive Rate'),
        y=alt.Y('TPR:Q', title='True Positive Rate'),
        color=alt.Color('AUC_label:N', title='Classes'),
        tooltip=['Classe', 'AUC', 'FPR', 'TPR']
    ).properties(
        title=titulo,
        width=700,
        height=400
    ).interactive()

    st.altair_chart(chart, use_container_width=True)

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
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        'Matriz Multinomial', 'Curva ROC Multinomial', 'Matriz XGBoost',
        'Curvas ROC XGBoost', 'Matriz Rede Neural', 'Curva ROC Rede Neural'
    ])

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
        plot_curvas_roc_altair(curva_roc_multinomial, "Curvas ROC - Regressão Multinomial")

    with tab3:
        st.subheader("Matriz de Confusão - XGBoost")
        st.write(matriz_xgboost)

    with tab4:
        st.subheader("Curvas ROC - XGBoost")
        plot_curvas_roc_altair(curva_roc_xgboost, "Curvas ROC - XGBoost")

    with tab5:
        st.subheader("Matriz de Confusão - Rede Neural")
        st.write(matriz_rede_neural)

    with tab6:
        st.subheader("Curvas ROC - Rede Neural")
        plot_curvas_roc_altair(curva_roc_rede_neural, "Curvas ROC - Rede Neural")

# Exibir o aplicativo
if __name__ == "__main__":
    show()
