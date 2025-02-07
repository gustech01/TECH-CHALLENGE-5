import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path
from PIL import Image
import os

# Função para carregar imagens com verificação de existência
@st.cache_resource
def carregar_imagem(nome_arquivo):
    caminho = os.path.join("imagens", nome_arquivo)  # Caminho da imagem
    if os.path.exists(caminho):
        return Image.open(caminho)
    else:
        st.error(f"Imagem '{nome_arquivo}' não encontrada. Verifique o caminho.")
        return None

@st.cache_data(show_spinner=True)
def carregar_dados(caminho):
    """Carrega um dataset CSV, normaliza os nomes das colunas e trata a coluna 'Classe'."""
    try:
        df = pd.read_csv(caminho)

        # Normalizar os nomes das colunas
        df.columns = df.columns.str.strip().str.replace(" ", "_").str.lower()

        # Se existir a coluna "unnamed:_0", renomeá-la para "classe"
        if "unnamed:_0" in df.columns:
            df.rename(columns={"unnamed:_0": "classe"}, inplace=True)

        # **Verificação Extra:** Se "classe" não existir, exibe um alerta e continua.
        if "classe" not in df.columns:
            st.warning(f"Atenção: O arquivo '{caminho}' não contém uma coluna 'Classe'.")
            return df

        # **Substituir índices numéricos pelos nomes das classes, se possível**
        if df["classe"].dtype == 'int64':  # Se os valores forem números
            if len(df.columns) > 1:
                df["classe"] = df["classe"].map(lambda x: df.columns[x+1] if x < len(df.columns)-1 else x)

        return df

    except FileNotFoundError:
        st.error(f"Arquivo não encontrado: {caminho}")
        return pd.DataFrame()

def plot_curvas_roc(curva_roc, titulo):
    """Plota as curvas ROC diretamente a partir dos valores do CSV."""
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
    """Função principal para exibir a interface do Streamlit."""

    # Layout do cabeçalho com as imagens
    col1, col2, col3 = st.columns([1, 3, 1])
    with col1:
        imagem_passos = carregar_imagem("Passos-magicos-icon-cor.png")
        if imagem_passos:
            st.image(imagem_passos, use_container_width=True)
    with col3:
        imagem_fiap = carregar_imagem("fiap.png")
        if imagem_fiap:
            st.image(imagem_fiap, use_container_width=True)

    # Título principal
    st.title('Análise de Modelos Preditivos')

    # Caminho dos datasets
    diretorio = "datasets/"
    base_path = Path(diretorio)

    if not base_path.is_dir():
        st.error(f"O diretório especificado '{diretorio}' não existe.")
        return

    # Dicionário com os arquivos de dados
    arquivos = {
        "Matriz de Confusão - Regressão Multinomial": "matriz_confusao_multinomial.csv",
        "Curvas ROC - Regressão Multinomial": "curvas_roc_multinomial.csv",
        "Matriz de Confusão - XGBoost": "matriz_confusao_xgboost.csv",
        "Curvas ROC - XGBoost": "curvas_roc_xgboost.csv",
        "Matriz de Confusão - Rede Neural": "matriz_confusao_rede_neural.csv",
        "Curvas ROC - Rede Neural": "curvas_roc_rede_neural.csv"
    }

    # Criando abas para cada conjunto de dados
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
                    with st.expander("📈 Gráfico da Curva ROC", expanded=True):
                        plot_curvas_roc(dados, titulo)

                    with st.expander("📊 Tabela de Dados da Curva ROC"):
                        st.write(dados)

            else:
                if dados.empty:
                    st.error(f"Os dados para {titulo} não puderam ser carregados.")
                else:
                    with st.expander("📊 Tabela de Dados da Matriz de Confusão", expanded=True):
                        st.write(dados)

# Executar o aplicativo
if __name__ == "__main__":
    show()
