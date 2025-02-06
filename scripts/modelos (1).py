# reallizar a instalação do -> pip install pandas numpy scipy statsmodels matplotlib prophet xgboost scikit-learn


import pandas as pd
import numpy as np
from scipy.stats import boxcox
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_percentage_error
from prophet import Prophet
import xgboost as xgb
import matplotlib.pyplot as plt
from datetime import timedelta

# Carregar dados
DadosTC4 = pd.read_csv("~/Downloads/Europe_Brent_Spot_Price_FOB.csv")
DadosTC4['Date'] = pd.to_datetime(DadosTC4['Date'], format='%m/%d/%Y')
DadosTC4 = DadosTC4.sort_values('Date')
DadosTC4 = DadosTC4.set_index('Date').asfreq('D').fillna(method='ffill')
DadosTC4['Value'] = DadosTC4['Value'].astype(float)

# Separando treino e teste
train_size = int(len(DadosTC4) * 0.95)
train_data = DadosTC4.iloc[:train_size]
test_data = DadosTC4.iloc[train_size:]

# Teste Dickey-Fuller
adf_test_result = adfuller(test_data['Value'])
print("ADF Statistic:", adf_test_result[0])
print("p-value:", adf_test_result[1])

# Série temporal do treino
TimeSeriesTreino = train_data['Value'].diff(365).dropna()

# Teste Dickey-Fuller na série diferenciada
adf_test_diff = adfuller(TimeSeriesTreino)
print("ADF Statistic (differenced):", adf_test_diff[0])
print("p-value (differenced):", adf_test_diff[1])

# Modelo ARIMA
arima_model = SARIMAX(train_data['Value'], seasonal_order=(1, 1, 1, 365), enforce_stationarity=False)
arima_fit = arima_model.fit(disp=False)
print(arima_fit.summary())

forecast_horizon = len(test_data)
arima_forecast = arima_fit.get_forecast(steps=forecast_horizon)
arima_predicted = arima_forecast.predicted_mean

# Plot ARIMA
plt.plot(test_data.index, test_data['Value'], label='Actual', color='blue', linewidth=2)
plt.plot(test_data.index, arima_predicted, label='Predicted', color='red', linewidth=2)
plt.legend(loc='upper left')
plt.title('Actual vs Predicted Values (ARIMA)')
plt.show()

# MAPE para ARIMA
mape_arima = mean_absolute_percentage_error(test_data['Value'], arima_predicted) * 100
print(f"MAPE (ARIMA): {mape_arima:.2f}%")

# Modelo Prophet
train_prophet = train_data.reset_index()[['Date', 'Value']].rename(columns={'Date': 'ds', 'Value': 'y'})
prophet_model = Prophet(seasonality_mode='additive', yearly_seasonality=True)
prophet_model.fit(train_prophet)

future = test_data.reset_index()[['Date']].rename(columns={'Date': 'ds'})
prophet_forecast = prophet_model.predict(future)

# Ajustar predições do Prophet
prophet_predicted = prophet_forecast['yhat']
mape_prophet = mean_absolute_percentage_error(test_data['Value'], prophet_predicted) * 100
print(f"MAPE (Prophet): {mape_prophet:.2f}%")

# Plot Prophet
plt.plot(test_data.index, test_data['Value'], label='Actual', color='blue', linewidth=2)
plt.plot(test_data.index, prophet_predicted, label='Predicted', color='red', linewidth=2)
plt.legend(loc='upper left')
plt.title('Actual vs Predicted Values (Prophet)')
plt.show()

# Modelo XGBoost
def create_lagged_features(df, lags):
    for lag in range(1, lags + 1):
        df[f'lag_{lag}'] = df['Value'].shift(lag)
    return df

lags = 5
train_lagged = create_lagged_features(train_data.copy(), lags=lags).dropna()
test_lagged = create_lagged_features(test_data.copy(), lags=lags).dropna()

train_matrix = train_lagged.drop(columns=['Value']).values
train_labels = train_lagged['Value'].values
test_matrix = test_lagged.drop(columns=['Value']).values
test_labels = test_lagged['Value'].values

xgb_model = xgb.XGBRegressor(n_estimators=4000, objective='reg:squarederror', verbosity=1)
xgb_model.fit(train_matrix, train_labels)

predictions_xgb = xgb_model.predict(test_matrix)

# MAPE para XGBoost
mape_xgb = mean_absolute_percentage_error(test_labels, predictions_xgb) * 100
print(f"MAPE (XGBoost): {mape_xgb:.2f}%")

# Plot XGBoost
plt.plot(test_data.index[lags:], test_labels, label='Actual', color='blue', linewidth=2)
plt.plot(test_data.index[lags:], predictions_xgb, label='Predicted', color='red', linewidth=2)
plt.legend(loc='upper left')
plt.title('Actual vs Predicted Values (XGBoost)')
plt.show()
