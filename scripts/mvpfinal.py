import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

# Função para carregar e processar os dados
@st.cache_data(show_spinner=True)
def carregar_dados():
    caminho = "PEDE_PASSOS_DATASET_FIAP.csv"
    df = pd.read_csv(caminho, delimiter=";")
    
    def preparar_dataset(df, ano):
        colunas = ["NOME", f"PEDRA_{ano}", f"IAA_{ano}", f"IEG_{ano}", f"IPS_{ano}", f"IDA_{ano}", f"IPP_{ano}", f"IPV_{ano}", f"IAN_{ano}"]
        df_ano = df[colunas].dropna()
        df_ano.columns = ["NOME", "Pedra", "IAA", "IEG", "IPS", "IDA", "IPP", "IPV", "IAN"]
        return df_ano
    
    df_final = pd.concat([preparar_dataset(df, ano) for ano in [2020, 2021, 2022]], ignore_index=True)
    df_final = df_final[df_final["Pedra"] != "#NULO!"]
    df_final.iloc[:, 2:] = df_final.iloc[:, 2:].apply(pd.to_numeric, errors='coerce')
    df_final.dropna(inplace=True)
    
    label_encoder = LabelEncoder()
    df_final["Pedra"] = label_encoder.fit_transform(df_final["Pedra"])
    
    return df_final, label_encoder

df_final, label_encoder = carregar_dados()
X = df_final.drop(columns=["NOME", "Pedra"])
y = df_final["Pedra"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Criar abas no Streamlit
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    'Matriz - Multinomial', 'Curva ROC - Multinomial', 
    'Matriz - XGBoost', 'Curva ROC - XGBoost', 
    'Matriz - Rede Neural', 'Curva ROC - Rede Neural'
])

# 🔹 Regressão Multinomial
modelo_multinomial = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=1000)
modelo_multinomial.fit(X_train, y_train)
y_pred_mult = modelo_multinomial.predict(X_test)

# 🔹 XGBoost
xgb_model = xgb.XGBClassifier(objective="multi:softmax", num_class=len(label_encoder.classes_), eval_metric="mlogloss")
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

# 🔹 Rede Neural
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = Sequential([
    Dense(8, activation="relu", input_shape=(X_train.shape[1],)),
    Dense(4, activation="relu"),
    Dense(len(label_encoder.classes_), activation="softmax")
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(X_train_scaled, y_train, epochs=50, batch_size=8, verbose=0)
y_pred_nn = np.argmax(model.predict(X_test_scaled), axis=1)

# 🔹 Função para exibir matriz de confusão
def plot_matriz(y_true, y_pred, titulo):
    cm = confusion_matrix(y_true, y_pred)
    fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                    labels=dict(x="Previsões", y="Valores Reais"))
    fig.update_layout(title=titulo)
    return fig

# 🔹 Função para exibir Curva ROC
def plot_roc(y_true, y_scores, titulo):
    y_bin = label_binarize(y_true, classes=np.arange(len(label_encoder.classes_)))
    fig = px.line(title=titulo)
    for i in range(len(label_encoder.classes_)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_scores[:, i])
        fig.add_scatter(x=fpr, y=tpr, mode='lines', name=f"Classe {label_encoder.classes_[i]}")
    return fig

# 🔹 Exibição dos gráficos
with tab1:
    st.subheader("Matriz de Confusão - Multinomial")
    st.plotly_chart(plot_matriz(y_test, y_pred_mult, "Matriz de Confusão - Multinomial"))
with tab3:
    st.subheader("Matriz de Confusão - XGBoost")
    st.plotly_chart(plot_matriz(y_test, y_pred_xgb, "Matriz de Confusão - XGBoost"))
with tab5:
    st.subheader("Matriz de Confusão - Rede Neural")
    st.plotly_chart(plot_matriz(y_test, y_pred_nn, "Matriz de Confusão - Rede Neural"))

# 🔹 Curvas ROC
with tab2:
    st.subheader("Curva ROC - Multinomial")
    st.plotly_chart(plot_roc(y_test, modelo_multinomial.predict_proba(X_test), "Curva ROC - Multinomial"))
with tab4:
    st.subheader("Curva ROC - XGBoost")
    st.plotly_chart(plot_roc(y_test, xgb_model.predict_proba(X_test), "Curva ROC - XGBoost"))
with tab6:
    st.subheader("Curva ROC - Rede Neural")
    st.plotly_chart(plot_roc(y_test, model.predict(X_test_scaled), "Curva ROC - Rede Neural"))
