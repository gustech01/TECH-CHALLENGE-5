import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path

@st.cache_data(show_spinner=True)
def carregar_dados(caminho):
    """Carrega um dataset CSV e retorna o DataFrame."""
    try:
        return pd.read_csv(caminho)
    except FileNotFoundError:
        st.error(f"Arquivo não encontrado: {caminho}")
        return pd.DataFrame()

def plot_curvas_roc(curva_roc, titulo):
    """
    Plota as curvas ROC diretamente a partir dos valores do CSV.
    Espera que o CSV contenha as colunas 'FPR', 'TPR', 'Classe' e 'AUC'.
    """
    if not all(col in curva_roc.columns for col in ['FPR', 'TPR', 'AUC', 'Classe']):
        st.error("O arquivo não contém as colunas necessárias: 'FPR', 'TPR', 'Classe', 'AUC'.")
        return

    curva_roc['Label'] = curva_roc['Classe'] + " (AUC = " + curva_roc['AUC'].round(6).astype(str) + ")"

    chart = alt.Chart(curva_roc).mark_line().encode(
        x=alt.X('FPR:Q', title='False Positive Rate'),
        y=alt.Y('TPR:Q', title='True Positive Rate'),
        color=alt.Color('Label:N', title='Classes (AUC)'),
        tooltip=['Classe', 'FPR', 'TPR', 'AUC']
    ).properties(
        title=titulo,
        width=700,
        height=400
    ).interactive()

    st.altair_chart(chart, use_container_width=True)

def show():
    st.title('Análise de Modelos Preditivos')
    
    # Caminho do diretório com os CSVs
    diretorio = "datasets/"  # Atualize para o diretório onde estão seus arquivos CSV
    base_path = Path(diretorio)
    
    if not base_path.is_dir():
        st.error(f"O diretório especificado '{diretorio}' não existe.")
        return
    
    # Arquivos esperados
    arquivos = {
        "Matriz de Confusão - Regressão Multinomial": "matriz_confusao_multinomial.csv",
        "Curvas ROC - Regressão Multinomial": "curvas_roc_multinomial.csv",
        "Matriz de Confusão - XGBoost": "matriz_confusao_xgboost.csv",
        "Curvas ROC - XGBoost": "curvas_roc_xgboost.csv",
        "Matriz de Confusão - Rede Neural": "matriz_confusao_rede_neural.csv",
        "Curvas ROC - Rede Neural": "curvas_roc_rede_neural.csv"
    }
    
    # Tabs no Streamlit
    abas = st.tabs(list(arquivos.keys()))

    for aba, (titulo, arquivo) in zip(abas, arquivos.items()):
        caminho_arquivo = base_path / arquivo
        dados = carregar_dados(caminho_arquivo)
        
        with aba:
            st.subheader(titulo)
            
            if "Curvas ROC" in titulo:  # Verifica se o arquivo é de curva ROC
                if dados.empty:
                    st.error(f"Os dados para {titulo} não puderam ser carregados.")
                else:
                    plot_curvas_roc(dados, titulo)
            else:  # Caso seja uma matriz de confusão
                if dados.empty:
                    st.error(f"Os dados para {titulo} não puderam ser carregados.")
                else:
                    st.write(dados)

# Executar o aplicativo
if __name__ == "__main__":
    show()
