import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize

# Certifique-se de que as bibliotecas estejam instaladas
try:
    import seaborn as sns
except ImportError:
    st.error("Seaborn não está instalado. Por favor, instale a biblioteca executando 'pip install seaborn'.")

# Carregando Dataset
file_path = "PEDE_PASSOS_DATASET_FIAP.csv"  # Ajuste para o caminho correto
df = pd.read_csv(file_path, delimiter=";")

# Criando datasets para cada ano e renomeando colunas
def preparar_dataset(df, ano):
    colunas = ["NOME", f"PEDRA_{ano}", f"IAA_{ano}", f"IEG_{ano}", f"IPS_{ano}", f"IDA_{ano}", f"IPP_{ano}", f"IPV_{ano}", f"IAN_{ano}"]
    df_ano = df[colunas].dropna()
    df_ano.columns = ["NOME", "Pedra", "IAA", "IEG", "IPS", "IDA", "IPP", "IPV", "IAN"]
    return df_ano

df_2020 = preparar_dataset(df, 2020)
df_2021 = preparar_dataset(df, 2021)
df_2022 = preparar_dataset(df, 2022)

df_final = pd.concat([df_2020, df_2021, df_2022], ignore_index=True)
df_final = df_final[df_final["Pedra"] != "#NULO!"]
num_cols = ["IAA", "IEG", "IPS", "IDA", "IPP", "IPV", "IAN"]
df_final[num_cols] = df_final[num_cols].apply(pd.to_numeric, errors='coerce')
df_final = df_final.dropna()

# Pré-processamento: Transformação das variáveis categóricas
label_encoder = LabelEncoder()
df_final["Pedra"] = label_encoder.fit_transform(df_final["Pedra"])

# Separação de dados para treino e teste
X = df_final.drop(columns=["NOME", "Pedra"])
y = df_final["Pedra"]

# Modelo de Regressão Multinomial
modelo_multinomial = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=1000)
modelo_multinomial.fit(X, y)

# Previsões e Avaliação do Modelo
y_pred = modelo_multinomial.predict(X)
cm = confusion_matrix(y, y_pred)

# Função para exibir a Matriz de Confusão
def plot_confusion_matrix(cm, classes, title='Matriz de Confusão', cmap=plt.cm.Blues):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xlabel("Previsões")
    plt.ylabel("Valores Reais")
    st.pyplot(plt)

# Função para exibir a Curva ROC
def plot_roc_curve(model, X, y, classes):
    y_bin = label_binarize(y, classes=np.unique(y))
    y_pred_proba = model.predict_proba(X)
    n_classes = y_bin.shape[1]
    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{classes[i]} (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("Taxa de Falsos Positivos")
    plt.ylabel("Taxa de Verdadeiros Positivos")
    plt.title("Curvas ROC - Regressão Multinomial")
    plt.legend()
    st.pyplot(plt)

# Interface Streamlit
st.title("Análise de Dados - ONG Passos Mágicos")
st.header("Modelo de Regressão Multinomial")

# Exibindo a Matriz de Confusão
st.subheader("Matriz de Confusão")
plot_confusion_matrix(cm, classes=label_encoder.classes_)

# Exibindo a Curva ROC
st.subheader("Curva ROC - Regressão Multinomial")
plot_roc_curve(modelo_multinomial, X, y, label_encoder.classes_)
