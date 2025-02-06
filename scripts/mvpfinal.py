pip install matplotlib
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from pathlib import Path


# Função para carregar dados
@st.cache_data(show_spinner=True)
def carregar_dados(caminho):
    try:
        return pd.read_csv(caminho)
    except FileNotFoundError:
        st.error(f"Arquivo não encontrado: {caminho}")
        return pd.DataFrame()

# Interface principal
def show():
    st.title('Matrizes de Confusão e Curvas ROC')
    
    # Criando abas (tabs)
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        'Matriz - Multinomial', 
        'Curva ROC - Multinomial', 
        'Matriz - XGBoost', 
        'Curva ROC - XGBoost', 
        'Matriz - Rede Neural', 
        'Curva ROC - Rede Neural'
    ])

    # 🔹 Matrizes de Confusão
    with tab1:
        st.subheader("Matriz de Confusão - Multinomial")
        df_multinomial = carregar_dados('dados/matriz_multinomial.csv')
        if not df_multinomial.empty:
            fig, ax = plt.subplots()
            sns.heatmap(df_multinomial, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Previsões")
            ax.set_ylabel("Valores Reais")
            st.pyplot(fig)
    
    with tab3:
        st.subheader("Matriz de Confusão - XGBoost")
        df_xgb = carregar_dados('dados/matriz_xgb.csv')
        if not df_xgb.empty:
            fig, ax = plt.subplots()
            sns.heatmap(df_xgb, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Previsões")
            ax.set_ylabel("Valores Reais")
            st.pyplot(fig)
    
    with tab5:
        st.subheader("Matriz de Confusão - Rede Neural")
        df_nn = carregar_dados('dados/matriz_nn.csv')
        if not df_nn.empty:
            fig, ax = plt.subplots()
            sns.heatmap(df_nn, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Previsões")
            ax.set_ylabel("Valores Reais")
            st.pyplot(fig)
    
    # 🔹 Curvas ROC
    def plot_roc_curve(df, modelo):
        if df.empty:
            st.error(f"Dados da Curva ROC ({modelo}) não encontrados!")
            return
        
        fig, ax = plt.subplots()
        for i in range(len(df.columns) // 2):  # Considerando que cada classe tem duas colunas: FPR e TPR
            fpr = df.iloc[:, i * 2]
            tpr = df.iloc[:, i * 2 + 1]
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"Classe {i} (AUC = {roc_auc:.2f})")
        
        ax.plot([0, 1], [0, 1], "k--")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"Curva ROC - {modelo}")
        ax.legend()
        st.pyplot(fig)
    
    with tab2:
        st.subheader("Curva ROC - Multinomial")
        df_roc_multinomial = carregar_dados('dados/roc_multinomial.csv')
        plot_roc_curve(df_roc_multinomial, "Multinomial")
    
    with tab4:
        st.subheader("Curva ROC - XGBoost")
        df_roc_xgb = carregar_dados('dados/roc_xgb.csv')
        plot_roc_curve(df_roc_xgb, "XGBoost")
    
    with tab6:
        st.subheader("Curva ROC - Rede Neural")
        df_roc_nn = carregar_dados('dados/roc_nn.csv')
        plot_roc_curve(df_roc_nn, "Rede Neural")

# Executar a interface no Streamlit
if __name__ == "__main__":
    show()
