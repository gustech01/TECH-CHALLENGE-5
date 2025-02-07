import streamlit as st
import pandas as pd
import altair as alt


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
    # Verificar se as colunas necessárias estão no DataFrame
    if not all(col in curva_roc.columns for col in ['FPR', 'TPR', 'Classe', 'AUC']):
        st.error("O arquivo não contém as colunas necessárias: 'FPR', 'TPR', 'Classe', 'AUC'.")
        return

    # Criar uma nova coluna para exibir as labels com o AUC
    curva_roc['Label'] = curva_roc['Classe'] + " (AUC = " + curva_roc['AUC'].round(2).astype(str) + ")"

    # Criar gráfico com Altair
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
    # Título
    st.title('Curvas ROC - Visualização Simples')

    # Upload de arquivo CSV
    uploaded_file = st.file_uploader("Carregue o arquivo CSV contendo as curvas ROC", type=["csv"])
    
    if uploaded_file is not None:
        curva_roc = carregar_dados(uploaded_file)
        st.write("Dados carregados:")
        st.dataframe(curva_roc)

        # Plotar gráfico
        st.subheader("Gráfico de Curvas ROC")
        plot_curvas_roc(curva_roc, "Curvas ROC")


# Executar o aplicativo
if __name__ == "__main__":
    show()
