import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path

@st.cache_data(show_spinner=True)
def carregar_dados(caminho):
    """Carrega um dataset CSV, normaliza os nomes das colunas e retorna o DataFrame."""
    try:
        st.write(f"Tentando carregar o arquivo: {caminho}")  # Debug para mostrar o caminho do arquivo
        df = pd.read_csv(caminho)
        st.write(f"Colunas carregadas: {df.columns.tolist()}")  # Debug para verificar as colunas
        df.columns = df.columns.str.strip().str.replace(" ", "_").str.lower()
        st.write("Colunas após normalização:", df.columns.tolist())  # Debug para verificar as colunas normalizadas
        return df
    except FileNotFoundError:
        st.error(f"Arquivo não encontrado: {caminho}")
        return pd.DataFrame()

def plot_curvas_roc(curva_roc, titulo):
    """
    Plota as curvas ROC diretamente a partir dos valores do CSV.
    Espera que o CSV contenha as colunas 'fpr', 'tpr', 'classe' e 'auc'.
    """
    # Verifica se todas as colunas necessárias estão presentes
    colunas_necessarias = {'fpr', 'tpr', 'auc', 'classe'}
    if not colunas_necessarias.issubset(curva_roc.columns):
        st.error(f"O arquivo não contém as colunas necessárias: {', '.join(colunas_necessarias)}.")
        st.write("Colunas encontradas no CSV:", curva_roc.columns.tolist())  # Debug para exibir as colunas
        st.write(curva_roc.head())  # Debug para exibir os dados carregados
        return

    # Criar uma nova coluna para labels formatados
    curva_roc['label'] = curva_roc['classe'] + " (AUC = " + curva_roc['auc'].round(6).astype(str) + ")"

    # Criar o gráfico Altair
    chart = alt.Chart(curva_roc).mark_line().encode(
        x=alt.X('fpr:Q', title='False Positive Rate'),
        y=alt.Y('tpr:Q', title='True Positive Rate'),
        color=alt.Color('label:N', title='Classes (AUC)'),
        tooltip=['classe', 'fpr', 'tpr', 'auc']
    ).properties(
        title=titulo,
        width=700,
        height=400
    ).interactive()

    # Exibir o gráfico no Streamlit
    st.altair_chart(chart, use_container_width=True)

def show():
    """Função principal para exibir as abas e carregar os dados."""
    st.title('Análise de Modelos Preditivos')
    
    # Diretório onde os arquivos CSV estão armazenados
    diretorio = "datasets/"
    base_path = Path(diretorio)
    
    if not base_path.is_dir():
        st.error(f"O diretório especificado '{diretorio}' não existe.")
        return
    
    # Dicionário de arquivos esperados
    arquivos = {
        "Matriz de Confusão - Regressão Multinomial": "matriz_confusao_multinomial.csv",
        "Curvas ROC - Regressão Multinomial": "curvas_roc_multinomial.csv",
        "Matriz de Confusão - XGBoost": "matriz_confusao_xgboost.csv",
        "Curvas ROC - XGBoost": "curvas_roc_xgboost.csv",
        "Matriz de Confusão - Rede Neural": "matriz_confusao_rede_neural.csv",
        "Curvas ROC - Rede Neural": "curvas_roc_rede_neural.csv"
    }
    
    # Criar abas no Streamlit
    abas = st.tabs(list(arquivos.keys()))

    for aba, (titulo, arquivo) in zip(abas, arquivos.items()):
        caminho_arquivo = base_path / arquivo
        dados = carregar_dados(caminho_arquivo)
        
        with aba:
            st.subheader(titulo)
            
            # Verificar se o arquivo é de Curvas ROC
            if "Curvas ROC" in titulo:
                st.write(f"Dados carregados para a aba: {titulo}")  # Debug
                st.write(dados.head())  # Exibe os primeiros dados carregados para verificação
                if dados.empty:
                    st.error(f"Os dados para {titulo} não puderam ser carregados.")
                else:
                    plot_curvas_roc(dados, titulo)
            else:
                # Mostrar tabela para matrizes de confusão
                if dados.empty:
                    st.error(f"Os dados para {titulo} não puderam ser carregados.")
                else:
                    st.write(dados)

# Executar o aplicativo
if __name__ == "__main__":
    show()
