import streamlit as st
import pandas as pd
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
    st.title('Matrizes e Modelos')

    # Layout do aplicativo
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(['Matriz - Multinomial', 'Curva ROC - Multinomial', 'Matriz - XGbBoost', 'Curva ROC - XGBoost', 'Matriz - Rede Neural', 'Curva ROC - Rede Neural]')

    # Leitura dos dados
    dados = carregar_dados("dataset/Europe_Brent_Spot_Price_FOB.csv")
    forecast = carregar_dados("dataset/xgboost_results.csv")

    if dados.empty or forecast.empty:
        st.error("Os arquivos de dataset não foram carregados corretamente.")
        return

    # Tratando dados históricos
    if 'Date' in dados.columns and 'Value' in dados.columns:
        dados['Date'] = pd.to_datetime(dados['Date'], errors='coerce')
        dados = dados[dados['Date'].between('2000-01-01', '2025-12-31')]
        dados.rename(columns={'Value': 'Real'}, inplace=True)
    else:
        st.error("Colunas 'Date' ou 'Value' ausentes no dataset histórico.")
        return

    # Tratando dados de previsões
    if 'Date' in forecast.columns and 'Predicted' in forecast.columns:
        forecast['Date'] = pd.to_datetime(forecast['Date'], errors='coerce')
        # Renomear a coluna 'Predicted' para 'α'
        forecast.rename(columns={'Predicted': 'Predito'}, inplace=True)
       
    else:
        st.error("Colunas 'Date' ou 'Predicted' ausentes no dataset de previsões.")
        return

    # Combinando dados históricos e forecast em um único DataFrame
    dados_comb = pd.merge(dados, forecast[['Date', 'Predito']], on='Date', how='outer')
    dados_comb = dados_comb.set_index('Date').sort_index()

  

# Exibir o aplicativo
if __name__ == "__main__":
    show()
