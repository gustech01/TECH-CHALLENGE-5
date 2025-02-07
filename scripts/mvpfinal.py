import streamlit as st
import pandas as pd
import altair as alt
from sklearn.metrics import auc


@st.cache_data(show_spinner=True)
def carregar_dados(caminho):
    """Carrega um dataset CSV, retorna o DataFrame ou erro."""
    try:
        return pd.read_csv(caminho, usecols=lambda column: column not in ["Unnamed: 0"])
    except FileNotFoundError:
        st.error(f"Arquivo não encontrado: {caminho}")
        return pd.DataFrame()


def calcular_auc(df):
    """
    Calcula o AUC para cada classe no DataFrame.
    """
    # Verifica se 'FPR' e 'TPR' existem para cálculo de AUC
    if 'FPR' not in df.columns or 'TPR' not in df.columns:
        st.error("As colunas 'FPR' ou 'TPR' não foram encontradas no DataFrame.")
        return df

    # Calcula o AUC para cada classe
    auc_values = df.groupby('Classe').apply(
        lambda group: auc(group['FPR'], group['TPR'])
    ).reset_index()
    auc_values.columns = ['Classe', 'AUC']

    # Junta os valores de AUC ao DataFrame original
    return df.merge(auc_values, on='Classe', how='left')


def plot_curvas_roc_altair(curva_roc, titulo):
    """
    Plota as curvas ROC usando a biblioteca Altair.
    """
    # Verificar se a coluna 'AUC' existe
    if 'AUC' not in curva_roc.columns:
        st.error("A coluna 'AUC' não foi encontrada no DataFrame. Verifique os dados carregados.")
        st.write("Colunas disponíveis no DataFrame:", curva_roc.columns.tolist())
        return

    # Criar rótulo para as legendas
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
    # Título
    st.title('Modelos Preditivos')

    # Layout do aplicativo
    tab1, tab2, tab3 = st.tabs(['Curva ROC Multinomial', 'Curvas ROC XGBoost', 'Curva ROC Rede Neural'])

    # Carregando as curvas ROC
    curva_roc_multinomial = carregar_dados("datasets/curvas_roc_multinomial.csv")
    curva_roc_xgboost = carregar_dados("datasets/curvas_roc_xgboost.csv")
    curva_roc_rede_neural = carregar_dados("datasets/curvas_roc_rede_neural.csv")

    # Verificação de carregamento
    if curva_roc_multinomial.empty or curva_roc_xgboost.empty or curva_roc_rede_neural.empty:
        st.error("Os arquivos de curvas ROC não foram carregados corretamente.")
        return

    # Calcular AUC
    curva_roc_multinomial = calcular_auc(curva_roc_multinomial)
    curva_roc_xgboost = calcular_auc(curva_roc_xgboost)
    curva_roc_rede_neural = calcular_auc(curva_roc_rede_neural)

    with tab1:
        st.subheader("Curvas ROC - Regressão Multinomial")
        plot_curvas_roc_altair(curva_roc_multinomial, "Curvas ROC - Regressão Multinomial")

    with tab2:
        st.subheader("Curvas ROC - XGBoost")
        plot_curvas_roc_altair(curva_roc_xgboost, "Curvas ROC - XGBoost")

    with tab3:
        st.subheader("Curvas ROC - Rede Neural")
        plot_curvas_roc_altair(curva_roc_rede_neural, "Curvas ROC - Rede Neural")


# Executar o aplicativo
if __name__ == "__main__":
    show()
