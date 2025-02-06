import streamlit as st
import pandas as pd
from pathlib import Path

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

def plot_curvas_roc(curva_roc, titulo):
    """Plota a curva ROC utilizando somente Streamlit."""
    df_combined = pd.DataFrame()
    legendas = {}

    for classe in curva_roc['Classe'].unique():
        dados_classe = curva_roc[curva_roc['Classe'] == classe]
        dados_classe = dados_classe.rename(columns={"TPR": f"TPR_{classe}"})
        auc_valor = ((dados_classe['FPR'] - dados_classe[f'TPR_{classe}']).abs().sum())  # Aproximação do AUC
        legendas[f"AUC_{classe}"] = f"{classe} (AUC = {auc_valor:.2f})"

        if df_combined.empty:
            df_combined = dados_classe[["FPR", f"TPR_{classe}"]]
        else:
            df_combined = pd.merge(df_combined, dados_classe[["FPR", f"TPR_{classe}"]], on="FPR", how="outer")

    # Gráfico menor
    st.line_chart(df_combined.set_index('FPR'), height=350, width=500)

    # Exibir legenda
    st.write("Legenda:")
    for key, value in legendas.items():
        st.write(value)

def renomear_matriz(matriz, classes):
    """Substitui índices numéricos pelos nomes das classes."""
    matriz.index = classes
    matriz.columns = classes
    return matriz

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
        'Matriz Multinomial', 'Curva ROC Multinomial', 
        'Matriz XGBoost', 'Curvas ROC XGBoost', 
        'Matriz Rede Neural', 'Curva Rede Neural'
    ])

    # Carregar datasets
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

    # Definir nomes das classes
    nomes_classes = ["Classe A", "Classe B", "Classe C", "Classe D", "Classe E"]

    with tab1:
        st.subheader("Matriz de Confusão - Regressão Multinomial")
        st.write(renomear_matriz(matriz_multinomial, nomes_classes))

    with tab2:
        st.subheader("Curvas ROC - Regressão Multinomial")
        plot_curvas_roc(curva_roc_multinomial, "Curvas ROC - Regressão Multinomial")

    with tab3:
        st.subheader("Matriz de Confusão - XGBoost")
        st.write(renomear_matriz(matriz_xgboost, nomes_classes))

    with tab4:
        st.subheader("Curvas ROC - XGBoost")
        plot_curvas_roc(curva_roc_xgboost, "Curvas ROC - XGBoost")

    with tab5:
        st.subheader("Matriz de Confusão - Rede Neural")
        st.write(renomear_matriz(matriz_rede_neural, nomes_classes))

    with tab6:
        st.subheader("Curvas ROC - Rede Neural")
        plot_curvas_roc(curva_roc_rede_neural, "Curvas ROC - Rede Neural")

# Exibir o aplicativo
if __name__ == "__main__":
    show()
