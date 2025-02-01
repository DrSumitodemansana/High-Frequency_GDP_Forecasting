import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import os
from src.__special__ import data_path


def actumen(file_path: str = data_path) -> None:
    # 1. Importar datos
    data = pd.read_excel(file_path, sheet_name="LECTURA")

    # 2. Definir funciones para modelos
    def fit_arima(data, order):
        model = ARIMA(data, order=order)
        results = model.fit()
        return results

    def fit_sarima(data, order, seasonal_order):
        model = SARIMAX(data, order=order, seasonal_order=seasonal_order)
        results = model.fit()
        return results

    # 3. Ajustar modelos y hacer pronósticos
    # Ejemplo para FEXP1
    results_exp1 = fit_arima(data['FEXP1'], order=(1, 1, 1))
    forecast_exp1 = results_exp1.forecast(steps=10)

    # Ejemplo para FEXP2 con SARIMA
    results_exp2 = fit_sarima(data['FEXP2'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    forecast_exp2 = results_exp2.forecast(steps=10)

    # 4. Exportar resultados
    forecast_exp1.to_excel("forecast_exp1.xlsx")
    forecast_exp2.to_excel("forecast_exp2.xlsx")


if __name__ == "__main__":
    file_name = os.path.join(data_path, "AFESP.xlsx")
    actumen(file_name)
