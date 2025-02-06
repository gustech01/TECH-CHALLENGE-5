import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

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
        fig = px.imshow(matriz_multinomial, 
                        x=matriz_multinomial.columns, 
                        y=matriz_multinomial.index, 
                        color_continuous_scale='Blues', 
                        text_auto=True, 
                        labels={'x': 'Predito', 'y': 'Real', 'color': 'Quantidade'})
        fig.update_layout(title="Matriz de Confusão - Regressão Multinomial", 
                          xaxis_title='Predito', 
                          yaxis_title='Real')
        st.plotly_chart(fig)

    with tab2:
        st.subheader("Curvas ROC - Regressão Multinomial")
        fig = px.line()
        for classe in curva_roc_multinomial['Classe'].unique():
            dados_classe = curva_roc_multinomial[curva_roc_multinomial['Classe'] == classe]
            fig.add_scatter(x=dados_classe['FPR'], y=dados_classe['TPR'], mode='lines', name=f"{classe} (AUC = {auc(dados_classe['FPR'], dados_classe['TPR']):.2f})")
        fig.add_scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), name='Linha de Referência')
        fig.update_layout(title='Curvas ROC - Regressão Multinomial', 
                          xaxis_title='False Positive Rate', 
                          yaxis_title='True Positive Rate')
        st.plotly_chart(fig)

    with tab3:
        st.subheader("Matriz de Confusão - XGBoost")
        fig = px.imshow(matriz_xgboost, 
                        x=matriz_xgboost.columns, 
                        y=matriz_xgboost.index, 
                        color_continuous_scale='Blues', 
                        text_auto=True, 
                        labels={'x': 'Predito', 'y': 'Real', 'color': 'Quantidade'})
        fig.update_layout(title="Matriz de Confusão - XGBoost", 
                          xaxis_title='Predito', 
                          yaxis_title='Real')
        st.plotly_chart(fig)

    with tab4:
        st.subheader("Curvas ROC - XGBoost")
        fig = px.line()
        for classe in curva_roc_xgboost['Classe'].unique():
            dados_classe = curva_roc_xgboost[curva_roc_xgboost['Classe'] == classe]
            fig.add_scatter(x=dados_classe['FPR'], y=dados_classe['TPR'], mode='lines', name=f"{classe} (AUC = {auc(dados_classe['FPR'], dados_classe['TPR']):.2f})")
        fig.add_scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), name='Linha de Referência')
        fig.update_layout(title='Curvas ROC - XGBoost', 
                          xaxis_title='False Positive Rate', 
                          yaxis_title='True Positive Rate')
        st.plotly_chart(fig)

    with tab5:
        st.subheader("Matriz de Confusão - Rede Neural")
        fig = px.imshow(matriz_rede_neural, 
                        x=matriz_rede_neural.columns, 
                        y=matriz_rede_neural.index, 
                        color_continuous_scale='Blues', 
                        text_auto=True, 
                        labels={'x': 'Predito', 'y': 'Real', 'color': 'Quantidade'})
        fig.update_layout(title="Matriz de Confusão - Rede Neural", 
                          xaxis_title='Predito', 
                          yaxis_title='Real')
        st.plotly_chart(fig)

    with tab6:
        st.subheader("Curvas ROC - Rede Neural")
        fig = px.line()
        for classe in curva_roc_rede_neural['Classe'].unique():
            dados_classe = curva_roc_rede_neural[curva_roc_rede_neural['Classe'] == classe]
            fig.add_scatter(x=dados_classe['FPR'], y=dados_classe['TPR'], mode='lines', name=f"{classe} (AUC = {auc(dados_classe['FPR'], dados_classe['TPR']):.2f})")
        fig.add_scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), name='Linha de Referência')
        fig.update_layout(title='Curvas ROC - Rede Neural', 
                          xaxis_title='False Positive Rate', 
                          yaxis_title='True Positive Rate')
        st.plotly_chart(fig)

# Exibir o aplicativo
if __name__ == "__main__":
    show()
