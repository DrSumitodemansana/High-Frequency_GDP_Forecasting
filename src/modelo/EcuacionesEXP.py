import statsmodels.api as sm
import pandas as pd
import os
from statsmodels.tsa.arima.model import ARIMA
from src.__special__ import indices_path

# Cargar los datos (debes reemplazar 'data.csv' con tu archivo real)

variables_arima_exportaciones = pd.read_csv(r'C:\Users\lucas\High-Frequency_GDP_Forecasting\src\modelo\actualizacion_mensual\variables_ecuaciones_arima\variables_arima_exportaciones.txt',delimiter='\t', parse_dates=['_date_'], index_col='_date_')
variables_arima_exportaciones = variables_arima_exportaciones.loc['2005-02-01':'2024-11-01']


def model_exp1(variables_arima_exportaciones):

    # Asegurar que el índice tenga frecuencia mensual
    variables_arima_exportaciones = variables_arima_exportaciones.asfreq('MS')

    # Definir las variables dependientes e independientes
    y = variables_arima_exportaciones['fexp1'].dropna()
    X = variables_arima_exportaciones[['pibuem']].iloc[1:]

    X = X.loc[y.index]  # Alinea X con los índices de y

    X = X[~X.isin([float('inf'), float('-inf')]).any(axis=1) & X.notna().all(axis=1)]

    model = ARIMA(y, exog=X, order=(1, 0, 0), seasonal_order=(1, 0, 0, 12))
    result = model.fit()  # maxiter=500, pgtol= 0.0001)
    return result

def model_exp2(variables_arima_exportaciones):

    # Asegurar que el índice tenga frecuencia mensual
    variables_arima_exportaciones = variables_arima_exportaciones.asfreq('MS')

    y = variables_arima_exportaciones['fexp2'].diff().dropna()
    X = variables_arima_exportaciones[['fvirus', 'fcexpor']].iloc[1:]

    X = X.loc[y.index]  # Alinea X con los índices de y

    X = X[~X.isin([float('inf'), float('-inf')]).any(axis=1) & X.notna().all(axis=1)]
    model = sm.tsa.statespace.SARIMAX(y, exog=X, order=([1, 0, 0, 1], 0, [0, 1, 0, 0, 0, 1]), seasonal_order=(0, 1, 2, 12))

    result = model.fit(warn_convergence=False, maxiter=20, pgtol=0.0001, method='lbfgs')
    return result

def model_exp3(variables_arima_exportaciones):

    # Asegurar que el índice tenga frecuencia mensual
    variables_arima_exportaciones = variables_arima_exportaciones.asfreq('MS')

    y = variables_arima_exportaciones['fexp3'].diff().dropna()
    X = variables_arima_exportaciones[['f202004', 'f202006']].iloc[1:]

    X = X.loc[y.index]  # Alinea X con los índices de y

    X = X[~X.isin([float('inf'), float('-inf')]).any(axis=1) & X.notna().all(axis=1)]

    model = ARIMA(y, exog=X, order=(2, 0, 1), seasonal_order=(0, 0, 1, 12))
    result = model.fit()  # maxiter=500, pgtol= 0.0001)
    return result

def model_exp4(variables_arima_exportaciones):

    # Asegurar que el índice tenga frecuencia mensual
    variables_arima_exportaciones = variables_arima_exportaciones.asfreq('MS')

    y = variables_arima_exportaciones['fexp4'].diff().dropna()
    X = variables_arima_exportaciones[['f202004', 'f202005']].iloc[1:]

    X = X.loc[y.index]  # Alinea X con los índices de y

    X = X[~X.isin([float('inf'), float('-inf')]).any(axis=1) & X.notna().all(axis=1)]

    model = sm.tsa.statespace.SARIMAX(y, exog=X, order=(2, 0, 2), seasonal_order=(0, 0, 1, 12))
    result = model.fit(maxiter=500, pgtol=0.0001)
    return result

def model_exp5(variables_arima_exportaciones):

    # Asegurar que el índice tenga frecuencia mensual
    variables_arima_exportaciones = variables_arima_exportaciones.asfreq('MS')

    y = variables_arima_exportaciones['fexp5'].diff().dropna()
    X = variables_arima_exportaciones[['f202005']].iloc[1:]

    X = X.loc[y.index]  # Alinea X con los índices de y

    X = X[~X.isin([float('inf'), float('-inf')]).any(axis=1) & X.notna().all(axis=1)]
    model = sm.tsa.ARIMA(y, exog=X, order=(1, 0, 0), seasonal_order=(0, 0, 1, 12), trend='n')
    result = model.fit()
    return result

# Ajustar los modelos
results = {
    'EXP1': model_exp1(),
    'EXP2': model_exp2(),
    'EXP3': model_exp3(),
    'EXP4': model_exp4(),
    'EXP5': model_exp5()
}

# Mostrar los resúmenes
for key, res in results.items():
    print(f'Model {key} Summary:\n', res.summary())


