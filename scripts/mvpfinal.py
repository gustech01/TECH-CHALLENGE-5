import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path

@st.cache_data(show_spinner=True)
def carregar_dados(caminho):
    """Carrega um dataset CSV, normaliza os nomes das colunas e retorna o DataFrame."""
    try:
        df = pd.read_csv(caminho)
        df.columns = df.columns.str.strip().str.replace(" ", "_").str.lower()
        return df
    except FileNotFoundError:
        st.error(f"Arquivo não encontrado: {caminho}")
        return pd.DataFrame()

def plot_curvas_roc(curva_roc, titulo):
    """
    Plota as curvas ROC diretamente a partir dos valores do CSV.
    Espera que o CSV contenha as colunas 'fpr', 'tpr', 'classe' e 'auc'.
    """
    colunas_necessarias = {'fpr', 'tpr', 'auc', 'classe'}
    if not colunas_necessarias.issubset(curva_roc.columns):
        st.error(f"O arquivo não contém as colunas necessárias: {', '.join(colunas_necessarias)}.")
        return

    curva_roc['label'] = curva_roc['classe'] + " (AUC = " + curva_roc['auc'].round(6).astype(str) + ")"

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

    st.altair_chart(chart, use_container_width=True)

def show():
    """Função principal para exibir as abas e organizar os elementos."""
    st.title('Análise de Modelos Preditivos')

    diretorio = "datasets/"
    base_path = Path(diretorio)

    if not base_path.is_dir():
        st.error(f"O diretório especificado '{diretorio}' não existe.")
        return

    arquivos = {
        "Matriz de Confusão - Regressão Multinomial": "matriz_confusao_multinomial.csv",
        "Curvas ROC - Regressão Multinomial": "curvas_roc_multinomial.csv",
        "Matriz de Confusão - XGBoost": "matriz_confusao_xgboost.csv",
        "Curvas ROC - XGBoost": "curvas_roc_xgboost.csv",
        "Matriz de Confusão - Rede Neural": "matriz_confusao_rede_neural.csv",
        "Curvas ROC - Rede Neural": "curvas_roc_rede_neural.csv"
    }

    abas = st.tabs(list(arquivos.keys()))

    for aba, (titulo, arquivo) in zip(abas, arquivos.items()):
        caminho_arquivo = base_path / arquivo
        dados = carregar_dados(caminho_arquivo)

        with aba:
            st.subheader(titulo)

            if "Curvas ROC" in titulo:
                if dados.empty:
                    st.error(f"Os dados para {titulo} não puderam ser carregados.")
                else:
                    # Gráfico separado em uma caixinha
                    with st.expander("Gráfico da Curva ROC", expanded=True):
                        plot_curvas_roc(dados, titulo)

                    # Tabela separada em outra caixinha
                    with st.expander("Tabela de Dados da Curva ROC"):
                        st.write(dados)

            else:
                if dados.empty:
                    st.error(f"Os dados para {titulo} não puderam ser carregados.")
                else:
                    # Matriz de Confusão separada em uma caixinha
                    with st.expander("Tabela de Dados da Matriz de Confusão", expanded=True):
                        st.write(dados)

# Executar o aplicativo
if __name__ == "__main__":
    show()
