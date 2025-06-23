import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
import warnings

# Ignorar todos los warnings
warnings.filterwarnings("ignore")

# ---------- CARGA DE DATOS ----------
variables_arima_inv_full = pd.read_csv(
    r'C:\Users\lucas\High-Frequency_GDP_Forecasting\src\UNION\FactoresPCA\factores_INV.txt',
    delimiter='\t', parse_dates=['_date_'], index_col='_date_'
).asfreq('MS')

# Cargar ambas columnas desde archivo
col_fexp1f = pd.read_csv(r'C:\Users\lucas\High-Frequency_GDP_Forecasting\src\UNION\Exogenas\fexp1f.csv', index_col=0, parse_dates=True)
col_f202004 = pd.read_csv(r'C:\Users\lucas\High-Frequency_GDP_Forecasting\src\UNION\Exogenas\f202004.csv', index_col=0, parse_dates=True)
# Cargar DataFrame base
variables_arima_inv_full = pd.read_csv(r'C:\Users\lucas\High-Frequency_GDP_Forecasting\src\UNION\FactoresPCA\factores_INV.txt',delimiter='\t', parse_dates=['_date_'], index_col='_date_').asfreq('MS')
# Unir ambas columnas al DataFrame original
variables_arima_inv_full = variables_arima_inv_full.join([col_fexp1f, col_f202004])

train_data = variables_arima_inv_full.loc['1996-01-01':'2024-12-01']
forecast_data = variables_arima_inv_full.loc['1996-01-01':'2025-12-01']


def run_model(df_forecast, numero, y_col, exog_cols=None, order=(1, 0, 0),
              seasonal_order=(0, 0, 0, 0), trend='n', start_date='2003-04-01',
              diff=False, verbose=True):
    # Y: aplicar .diff() SOLO si diff=True
    if diff:
        y_full = train_data[y_col].diff()
        y = y_full.loc[start_date:]
    else:
        y = train_data[y_col].loc[start_date:]
    y = y.dropna()

    # X: NO aplicar diff() aunque y esté diferenciada
    X = None
    if exog_cols:
        X_raw = train_data[exog_cols].loc[start_date:]
        X = X_raw.loc[y.index]
        X = X[~X.isin([float('inf'), float('-inf')]).any(axis=1) & X.notna().all(axis=1)]

        # Alinear y con X si por limpieza alguna fila desapareció
        y = y.loc[X.index]

    model = ARIMA(y, exog=X, order=order, seasonal_order=seasonal_order, trend=trend)
    result = model.fit()

    # if verbose:
    #     print(f"\n--- Modelo finv{numero} ---")
    #     print(result.summary())

    # Forecast
    steps = 12
    future_dates = pd.date_range(start='2025-01-01', periods=steps, freq='MS')
    X_future = df_forecast.loc[future_dates, exog_cols] if exog_cols else None

    forecast = result.forecast(steps=steps, exog=X_future).to_frame(name='predicted_mean')

    if diff:
        last_val = train_data[y_col].loc['2024-12-01']
        forecast['predicted_mean'] = forecast['predicted_mean'].cumsum() + last_val

    col_forecast = f'finv{numero}f'
    df_forecast = df_forecast.copy()
    df_forecast[col_forecast] = df_forecast[y_col]
    df_forecast.loc[future_dates, col_forecast] = forecast['predicted_mean']

    serie_completa = df_forecast[col_forecast].loc['2003-03-01':'2025-12-01']
    _, hp_cycle = sm.tsa.filters.hpfilter(serie_completa, lamb=1)

    # Nueva salida: la serie sin HP
    serie_sin_hp = serie_completa.rename(f'FINV{numero}F')

    return hp_cycle.rename(f'FINV{numero}F_H'), serie_sin_hp, df_forecast


# ---------- EJECUCIÓN DE LOS MODELOS ----------
forecast_data_mod = forecast_data.copy()

FINV1F_H, FINV1F, forecast_data_mod = run_model(
    forecast_data_mod,
    numero=1,
    y_col='finv1',
    exog_cols=['fexp1f'],
    order=(1, 0, 1),
    seasonal_order=(1, 0, 0, 12),
    trend='n',
    start_date='2005-03-01',
    diff=False
)

FINV2F_H, FINV2F, forecast_data_mod = run_model(
    forecast_data_mod,
    numero=2,
    y_col='finv2',
    exog_cols=None,
    order=(2, 0, [0] * 11 + [1]),
    seasonal_order=(0, 0, 0, 10),
    trend='n',
    start_date='2003-03-01',
    diff=True
)

FINV3F_H, FINV3F, forecast_data_mod = run_model(
    forecast_data_mod,
    numero=3,
    y_col='finv3',
    exog_cols=['f202004'],
    order=(2, 0, 0),
    seasonal_order=(1, 0, 0, 12),
    trend='n',
    start_date='2003-03-01',
    diff=True
)

FINV4F_H, FINV4F, forecast_data_mod = run_model(
    forecast_data_mod,
    numero=4,
    y_col='finv4',
    exog_cols=None,
    order=(2, 0, 0),
    seasonal_order=(1, 0, 0, 12),
    trend='n',
    start_date='2003-03-01',
    diff=True
)

FINV5F_H, FINV5F, forecast_data_mod = run_model(
    forecast_data_mod,
    numero=5,
    y_col='finv5',
    exog_cols=None,
    order=(2, 0, [0] * 11 + [1]),
    seasonal_order=(0, 0, 0, 0),
    trend='n',
    start_date='2003-03-01',
    diff=False
)

# ---------- GUARDAR SERIES HP ----------
output_path = r'C:\Users\lucas\MAF_ESP\Forecast_INV_HP.csv'
hp_series_inv = pd.concat([
    FINV1F_H,
    FINV2F_H,
    FINV3F_H,
    FINV4F_H,
    FINV5F_H
], axis=1)
hp_series_inv.to_csv(output_path)

# Guardar series sin filtro HP
s_series_inv = pd.concat([FINV1F,FINV2F,FINV3F,FINV4F,FINV5F], axis=1)
s_series_inv.to_csv(r'C:\Users\lucas\High-Frequency_GDP_Forecasting\src\UNION\Exogenas_sin_HP\INV_sin_HP.csv')

# ESTA PERFECTO

# ---------- CARGA DE DATOS ----------
variables_arima_ind_full = pd.read_csv(
    r'C:\Users\lucas\High-Frequency_GDP_Forecasting\src\UNION\FactoresPCA\factores_IND.txt',
    delimiter='\t', parse_dates=['_date_'], index_col='_date_'
).asfreq('MS')

# Cargar ambas columnas desde archivo
col_fexp1f = pd.read_csv(r'C:\Users\lucas\High-Frequency_GDP_Forecasting\src\UNION\Exogenas\fexp1f.csv', index_col=0, parse_dates=True)
col_fexp2f = pd.read_csv(r'C:\Users\lucas\High-Frequency_GDP_Forecasting\src\UNION\Exogenas\fexp2f.csv', index_col=0, parse_dates=True)
col_finv1f = pd.read_csv(r'C:\Users\lucas\High-Frequency_GDP_Forecasting\src\UNION\Exogenas\finv1f.csv', index_col=0, parse_dates=True)
col_f202003 = pd.read_csv(r'C:\Users\lucas\High-Frequency_GDP_Forecasting\src\UNION\Exogenas\f202003.csv', index_col=0, parse_dates=True)
col_f202004 = pd.read_csv(r'C:\Users\lucas\High-Frequency_GDP_Forecasting\src\UNION\Exogenas\f202004.csv', index_col=0, parse_dates=True)
col_f202005 = pd.read_csv(r'C:\Users\lucas\High-Frequency_GDP_Forecasting\src\UNION\Exogenas\f202005.csv', index_col=0, parse_dates=True)
col_f202006 = pd.read_csv(r'C:\Users\lucas\High-Frequency_GDP_Forecasting\src\UNION\Exogenas\f202006.csv', index_col=0, parse_dates=True)
# Cargar DataFrame base
variables_arima_ind_full = pd.read_csv(r'C:\Users\lucas\High-Frequency_GDP_Forecasting\src\UNION\FactoresPCA\factores_IND.txt',delimiter='\t', parse_dates=['_date_'], index_col='_date_').asfreq('MS')
# Unir ambas columnas al DataFrame original
variables_arima_ind_full = variables_arima_ind_full.join([col_fexp1f, col_f202004, col_fexp2f, col_finv1f, col_f202003, col_f202005, col_f202006])

train_data_ind = variables_arima_ind_full.loc['1996-01-01':'2024-12-01']
forecast_data_ind = variables_arima_ind_full.loc['1996-01-01':'2025-12-01']


def run_model_ind(df_forecast, numero, y_col, exog_cols=None, order=(1, 0, 0),
                  seasonal_order=(0, 0, 0, 0), trend='n', start_date='2003-04-01',
                  diff=False, diff_exog=False, verbose=True):
    # --- Preparar y ---
    if diff:
        y_full = train_data_ind[y_col].diff()
        y = y_full.loc[start_date:]
    else:
        y = train_data_ind[y_col].loc[start_date:]
    y = y.dropna()

    # --- Preparar X ---
    X = None
    if exog_cols:
        X_raw = train_data_ind[exog_cols]
        if diff_exog:
            X_raw = X_raw.diff()
        X = X_raw.loc[start_date:]
        X = X.loc[y.index]
        X = X[~X.isin([float('inf'), float('-inf')]).any(axis=1) & X.notna().all(axis=1)]
        y = y.loc[X.index]

    # --- Entrenar modelo ---
    model = ARIMA(y, exog=X, order=order, seasonal_order=seasonal_order, trend=trend)
    result = model.fit()

    # if verbose:
    #     print(f"\n--- Modelo find{numero} ---")
    #     print(result.summary())

    # --- Forecast (12 meses) ---
    steps = 12
    future_dates = pd.date_range(start='2025-01-01', periods=steps, freq='MS')

    X_future = None
    if exog_cols:
        if diff_exog:
            # Necesita incluir 2024-12 para poder calcular .diff() con 2025-01 en adelante
            extended_dates = pd.date_range(start='2024-12-01', periods=steps + 1, freq='MS')
            X_future_raw = df_forecast.loc[extended_dates, exog_cols]
            X_future = X_future_raw.diff().dropna()
        else:
            X_future = df_forecast.loc[future_dates, exog_cols]

    forecast = result.forecast(steps=len(future_dates), exog=X_future).to_frame(name='predicted_mean')

    # Si Y fue diferenciada, reconstituir nivel original
    if diff:
        last_val = train_data_ind[y_col].loc['2024-12-01']
        forecast['predicted_mean'] = forecast['predicted_mean'].cumsum() + last_val

    # --- Inyectar predicción en dataframe completo ---
    col_forecast = f'find{numero}f'
    df_forecast = df_forecast.copy()
    df_forecast[col_forecast] = df_forecast[y_col]
    df_forecast.loc[forecast.index, col_forecast] = forecast['predicted_mean']

    # --- Filtrar y aplicar HP ---
    serie_completa = df_forecast[col_forecast].loc['2003-03-01':'2025-12-01']
    _, hp_cycle = sm.tsa.filters.hpfilter(serie_completa, lamb=1)

    # Nueva salida: la serie sin HP
    serie_sin_hp = serie_completa.rename(f'FIND{numero}F')

    return hp_cycle.rename(f'FIND{numero}F_H'), serie_sin_hp, df_forecast


# ---------- EJECUCIÓN DE LOS MODELOS ----------
forecast_data_ind_mod = forecast_data_ind.copy()

FIND1F_H, FIND1F, forecast_data_ind_mod = run_model_ind(
    forecast_data_ind_mod,
    numero=1,
    y_col='find1',
    exog_cols=['fexp1f', 'finv1f'],
    order=(2, 0, [0, 0, 1]),
    seasonal_order=(0, 0, 1, 12),
    trend='n',
    start_date='2005-03-01',
    diff=False
)

FIND2F_H, FIND2F, forecast_data_ind_mod = run_model_ind(
    forecast_data_ind_mod,
    numero=2,
    y_col='find2',
    exog_cols=['fexp2f', 'finv1f'],
    order=(0, 0, [1, 0, 1]),
    seasonal_order=(0, 0, 1, 12),
    trend='n',
    start_date='2005-04-01',
    diff=True,
    diff_exog=True
)

FIND3F_H, FIND3F, forecast_data_ind_mod = run_model_ind(
    forecast_data_ind_mod,
    numero=3,
    y_col='find3',
    exog_cols=['f202004', 'f202003'],
    order=(2, 0, [0, 0, 1]),
    seasonal_order=(0, 0, 1, 12),
    trend='n',
    start_date='2003-04-01',
    diff=True
)

FIND4F_H, FIND4F, forecast_data_ind_mod = run_model_ind(
    forecast_data_ind_mod,
    numero=4,
    y_col='find4',
    exog_cols=['f202004', 'f202006'],
    order=([0, 1], 0, [0, 1, 0]),
    seasonal_order=(0, 0, 1, 12),
    trend='n',
    start_date='2003-04-01',
    diff=True
)

FIND5F_H, FIND5F, forecast_data_ind_mod = run_model_ind(
    forecast_data_ind_mod,
    numero=5,
    y_col='find5',
    exog_cols=None,
    order=(2, 0, 2),
    seasonal_order=(0, 0, 1, 12),
    trend='n',
    start_date='2003-04-01',
    diff=True
)

FIND6F_H, FIND6F, forecast_data_ind_mod = run_model_ind(
    forecast_data_ind_mod,
    numero=6,
    y_col='find6',
    exog_cols=['f202005'],
    order=(1, 0, [0, 1]),
    seasonal_order=(0, 0, 1, 12),
    trend='n',
    start_date='2003-04-01',
    diff=True
)

# ---------- GUARDAR SERIES HP ----------
output_path = r'C:\Users\lucas\MAF_ESP\Forecast_IND_HP.csv'
hp_series_ind = pd.concat([
    FIND1F_H,
    FIND2F_H,
    FIND3F_H,
    FIND4F_H,
    FIND5F_H,
    FIND6F_H
], axis=1)
hp_series_ind.to_csv(output_path)

# Guardar series sin filtro HP
s_series_ind = pd.concat([FIND1F,FIND2F,FIND3F,FIND4F,FIND5F,FIND6F], axis=1)
s_series_ind.to_csv(r'C:\Users\lucas\High-Frequency_GDP_Forecasting\src\UNION\Exogenas_sin_HP\IND_sin_HP.csv')

# ESTA PERFECTO

# ---------- CARGA DE DATOS ----------
variables_arima_cpu_full = pd.read_csv(
    r'C:\Users\lucas\High-Frequency_GDP_Forecasting\src\UNION\FactoresPCA\factores_CPU.txt',
    delimiter='\t', parse_dates=['_date_'], index_col='_date_'
).asfreq('MS')

train_data_cpu = variables_arima_cpu_full.loc['1996-01-01':'2024-12-01']
forecast_data_cpu = variables_arima_cpu_full.loc['1996-01-01':'2025-12-01']


def run_model_cpu(df_forecast, numero, y_col, exog_cols=None, order=(1, 0, 0),
                  seasonal_order=(0, 0, 0, 0), trend='n', start_date='2003-04-01',
                  diff=False, diff_exog=False, verbose=True):
    # Preparar y
    if diff:
        y_full = train_data_cpu[y_col].diff()
        y = y_full.loc[start_date:]
    else:
        y = train_data_cpu[y_col].loc[start_date:]
    y = y.dropna()

    # Preparar X
    X = None
    if exog_cols:
        X_raw = train_data_cpu[exog_cols]
        if diff_exog:
            X_raw = X_raw.diff()
        X = X_raw.loc[start_date:]
        X = X.loc[y.index]
        X = X[~X.isin([float('inf'), float('-inf')]).any(axis=1) & X.notna().all(axis=1)]
        y = y.loc[X.index]

    # Entrenar modelo
    model = ARIMA(y, exog=X, order=order, seasonal_order=seasonal_order, trend=trend)
    result = model.fit()

    # if verbose:
    #     print(f"\n--- Modelo fcpu{numero} ---")
    #     print(result.summary())

    # Forecast
    steps = 12
    future_dates = pd.date_range(start='2025-01-01', periods=steps, freq='MS')

    X_future = None
    if exog_cols:
        if diff_exog:
            extended_dates = pd.date_range(start='2024-12-01', periods=steps + 1, freq='MS')
            X_future_raw = df_forecast.loc[extended_dates, exog_cols]
            X_future = X_future_raw.diff().dropna()
        else:
            X_future = df_forecast.loc[future_dates, exog_cols]

    forecast = result.forecast(steps=len(future_dates), exog=X_future).to_frame(name='predicted_mean')

    # Reconstruir valores si y fue diferenciada
    if diff:
        last_val = train_data_cpu[y_col].loc['2024-12-01']
        forecast['predicted_mean'] = forecast['predicted_mean'].cumsum() + last_val

    # Guardar predicción en df completo
    col_forecast = f'fcpu{numero}f'
    df_forecast = df_forecast.copy()
    df_forecast[col_forecast] = df_forecast[y_col]
    df_forecast.loc[forecast.index, col_forecast] = forecast['predicted_mean']

    # Aplicar filtro HP
    serie_completa = df_forecast[col_forecast].loc['2003-03-01':'2025-12-01']
    _, hp_cycle = sm.tsa.filters.hpfilter(serie_completa, lamb=1)

    # Nueva salida: la serie sin HP
    serie_sin_hp = serie_completa.rename(f'FCPU{numero}F')

    return hp_cycle.rename(f'FCPU{numero}F_H'), serie_sin_hp, df_forecast


# ---------- EJECUCIÓN DE LOS MODELOS ----------
forecast_data_cpu_mod = forecast_data_cpu.copy()

FCPU1F_H, FCPU1F, forecast_data_cpu_mod = run_model_cpu(
    forecast_data_cpu_mod,
    numero=1,
    y_col='fcpu1',
    order=(1, 0, [0, 0, 1]),
    seasonal_order=(0, 0, 1, 12),
    trend='n',
    start_date='2003-04-01',
    diff=True
)

FCPU2F_H, FCPU2F, forecast_data_cpu_mod = run_model_cpu(
    forecast_data_cpu_mod,
    numero=2,
    y_col='fcpu2',
    order=(1, 0, [0, 0, 1]),
    seasonal_order=(0, 0, 1, 12),
    trend='n',
    start_date='2003-04-01',
    diff=True
)

FCPU3F_H, FCPU3F, forecast_data_cpu_mod = run_model_cpu(
    forecast_data_cpu_mod,
    numero=3,
    y_col='fcpu3',
    order=(1, 0, [1, 0, 1]),
    seasonal_order=(0, 0, 1, 12),
    trend='n',
    start_date='2003-04-01',
    diff=True
)

# ---------- GUARDAR SERIES HP ----------
output_path = r'C:\Users\lucas\MAF_ESP\Forecast_CPU_HP.csv'
hp_series_cpu = pd.concat([
    FCPU1F_H,
    FCPU2F_H,
    FCPU3F_H,
], axis=1)
hp_series_cpu.to_csv(output_path)

# Guardar series sin filtro HP
s_series_cpu = pd.concat([FCPU1F, FCPU2F, FCPU3F], axis=1)
s_series_cpu.to_csv(r'C:\Users\lucas\High-Frequency_GDP_Forecasting\src\UNION\Exogenas_sin_HP\CPU_sin_HP.csv')

#ESTA PERFECTO

# ---------- CARGA DE DATOS ----------
variables_arima_agri_full = pd.read_csv(
    r'C:\Users\lucas\High-Frequency_GDP_Forecasting\src\UNION\FactoresPCA\factores_AGR.txt',
    delimiter='\t', parse_dates=['_date_'], index_col='_date_'
).asfreq('MS')

train_data_agri = variables_arima_agri_full.loc['1996-01-01':'2024-12-01']
forecast_data_agri = variables_arima_agri_full.loc['1996-01-01':'2025-12-01']


def run_model_agri(df_forecast, numero, y_col, exog_cols=None, order=(1, 0, 0),
                   seasonal_order=(0, 0, 0, 0), trend='n', start_date='2003-04-01',
                   diff=False, diff_exog=False, verbose=True):
    if diff:
        y_full = train_data_agri[y_col].diff()
        y = y_full.loc[start_date:]
    else:
        y = train_data_agri[y_col].loc[start_date:]
    y = y.dropna()

    X = None
    if exog_cols:
        X_raw = train_data_agri[exog_cols]
        if diff_exog:
            X_raw = X_raw.diff()
        X = X_raw.loc[start_date:]
        X = X.loc[y.index]
        X = X[~X.isin([float('inf'), float('-inf')]).any(axis=1) & X.notna().all(axis=1)]
        y = y.loc[X.index]

    model = ARIMA(y, exog=X, order=order, seasonal_order=seasonal_order, trend=trend)
    result = model.fit()

    # if verbose:
    #     print(f"\n--- Modelo fagr{numero} ---")
    #     print(result.summary())

    steps = 12
    future_dates = pd.date_range(start='2025-01-01', periods=steps, freq='MS')

    X_future = None
    if exog_cols:
        if diff_exog:
            extended_dates = pd.date_range(start='2024-12-01', periods=steps + 1, freq='MS')
            X_future_raw = df_forecast.loc[extended_dates, exog_cols]
            X_future = X_future_raw.diff().dropna()
        else:
            X_future = df_forecast.loc[future_dates, exog_cols]

    forecast = result.forecast(steps=len(future_dates), exog=X_future).to_frame(name='predicted_mean')

    if diff:
        last_val = train_data_agri[y_col].loc['2024-12-01']
        forecast['predicted_mean'] = forecast['predicted_mean'].cumsum() + last_val

    col_forecast = f'fagr{numero}f'
    df_forecast = df_forecast.copy()
    df_forecast[col_forecast] = df_forecast[y_col]
    df_forecast.loc[forecast.index, col_forecast] = forecast['predicted_mean']

    serie_completa = df_forecast[col_forecast].loc['2006-03-01':'2025-12-01']
    _, hp_cycle = sm.tsa.filters.hpfilter(serie_completa, lamb=1)

    # Nueva salida: la serie sin HP
    serie_sin_hp = serie_completa.rename(f'FAG{numero}F')

    return hp_cycle.rename(f'FAGR{numero}F_H'),serie_sin_hp, df_forecast


# ---------- EJECUCIÓN DE LOS MODELOS ----------
forecast_data_agri_mod = forecast_data_agri.copy()

FAGR1F_H, FAGR1F, forecast_data_agri_mod = run_model_agri(
    forecast_data_agri_mod,
    numero=1,
    y_col='fagr1',
    order=(2, 0, [0, 0, 1]),
    seasonal_order=(0, 0, 1, 12),
    trend='n',
    start_date='2006-03-01',
    diff=False
)

FAGR2F_H, FAGR2F, forecast_data_agri_mod = run_model_agri(
    forecast_data_agri_mod,
    numero=2,
    y_col='fagr2',
    order=(4, 0, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
    seasonal_order=(0, 0, 0, 0),
    trend='n',
    start_date='2006-04-01',
    diff=True
)

FAGR3F_H, FAGR3F, forecast_data_agri_mod = run_model_agri(
    forecast_data_agri_mod,
    numero=3,
    y_col='fagr3',
    order=(1, 0, [0, 0, 1]),
    seasonal_order=(0, 0, 1, 12),
    trend='n',
    start_date='2006-04-01',
    diff=True
)

FAGR4F_H, FAGR4F, forecast_data_agri_mod = run_model_agri(
    forecast_data_agri_mod,
    numero=4,
    y_col='fagr4',
    order=(3, 0, 0),
    seasonal_order=(1, 0, 0, 12),
    trend='n',
    start_date='2006-04-01',
    diff=True
)

FAGR5F_H, FAGR5F, forecast_data_agri_mod = run_model_agri(
    forecast_data_agri_mod,
    numero=5,
    y_col='fagr5',
    order=([0, 0, 0, 1], 0, [1, 0, 1]),
    seasonal_order=(0, 0, 1, 12),
    trend='n',
    start_date='2006-04-01',
    diff=True
)

# ---------- GUARDAR SERIES HP ----------
output_path_agri = r'C:\Users\lucas\MAF_ESP\Forecast_AGR_HP.csv'
hp_series_agri = pd.concat([
    FAGR1F_H,
    FAGR2F_H,
    FAGR3F_H,
    FAGR4F_H,
    FAGR5F_H
], axis=1)
hp_series_agri.to_csv(output_path_agri)

# Guardar series sin filtro HP
s_series_agri = pd.concat([FAGR1F, FAGR2F, FAGR3F, FAGR4F, FAGR5F], axis=1)
s_series_agri.to_csv(r'C:\Users\lucas\High-Frequency_GDP_Forecasting\src\UNION\Exogenas_sin_HP\AGR_sin_HP.csv')

# ESTA PERFECTO MENOS CPR4

# ---------- CARGA DE DATOS ----------
variables_arima_cpr_full = pd.read_csv(
    r'C:\Users\lucas\High-Frequency_GDP_Forecasting\src\UNION\FactoresPCA\factores_CP.txt',
    delimiter='\t', parse_dates=['_date_'], index_col='_date_'
).asfreq('MS')

# Cargar ambas columnas desde archivo
col_find1f = pd.read_csv(r'C:\Users\lucas\High-Frequency_GDP_Forecasting\src\UNION\Exogenas\find1f.csv', index_col=0, parse_dates=True)
col_fcon1f = pd.read_csv(r'C:\Users\lucas\High-Frequency_GDP_Forecasting\src\UNION\Exogenas\fcon1f.csv', index_col=0, parse_dates=True)
col_find2f = pd.read_csv(r'C:\Users\lucas\High-Frequency_GDP_Forecasting\src\UNION\Exogenas\find2f.csv', index_col=0, parse_dates=True)
col_find3f = pd.read_csv(r'C:\Users\lucas\High-Frequency_GDP_Forecasting\src\UNION\Exogenas\find3f.csv', index_col=0, parse_dates=True)
col_fcon3f = pd.read_csv(r'C:\Users\lucas\High-Frequency_GDP_Forecasting\src\UNION\Exogenas\fcon3f.csv', index_col=0, parse_dates=True)
col_f202005 = pd.read_csv(r'C:\Users\lucas\High-Frequency_GDP_Forecasting\src\UNION\Exogenas\f202005.csv', index_col=0, parse_dates=True)
# Cargar DataFrame base
variables_arima_cpr_full = pd.read_csv(r'C:\Users\lucas\High-Frequency_GDP_Forecasting\src\UNION\FactoresPCA\factores_CP.txt',delimiter='\t', parse_dates=['_date_'], index_col='_date_').asfreq('MS')
# Unir ambas columnas al DataFrame original
variables_arima_cpr_full = variables_arima_cpr_full.join([col_f202005, col_find1f, col_fcon1f, col_find2f, col_find3f, col_fcon3f])

train_data_cpr = variables_arima_cpr_full.loc['1996-01-01':'2024-12-01']
forecast_data_cpr = variables_arima_cpr_full.loc['1996-01-01':'2025-12-01']


def run_model_cpr(df_forecast, numero, y_col, exog_cols=None, order=(1, 0, 0),
                  seasonal_order=(0, 0, 0, 0), trend='n', start_date='2003-04-01',
                  diff=False, diff_exog=False, verbose=True):
    # --- Preparar y ---
    if diff:
        y_full = train_data_cpr[y_col].diff()
        y = y_full.loc[start_date:]
    else:
        y = train_data_cpr[y_col].loc[start_date:]
    y = y.dropna()

    # --- Preparar X ---
    X = None
    if exog_cols:
        X_raw = train_data_cpr[exog_cols]
        if diff_exog:
            X_raw = X_raw.diff()
        X = X_raw.loc[start_date:]
        X = X.loc[y.index]
        X = X[~X.isin([float('inf'), float('-inf')]).any(axis=1) & X.notna().all(axis=1)]
        y = y.loc[X.index]

    # --- Entrenar modelo ---
    model = ARIMA(y, exog=X, order=order, seasonal_order=seasonal_order, trend=trend)
    result = model.fit()

    # if verbose:
    #     print(f"\n--- Modelo fcp{numero} ---")
    #     print(result.summary())

    # --- Forecast (12 meses) ---
    steps = 12
    future_dates = pd.date_range(start='2025-01-01', periods=steps, freq='MS')

    X_future = None
    if exog_cols:
        if diff_exog:
            extended_dates = pd.date_range(start='2024-12-01', periods=steps + 1, freq='MS')
            X_future_raw = df_forecast.loc[extended_dates, exog_cols]
            X_future = X_future_raw.diff().dropna()
        else:
            X_future = df_forecast.loc[future_dates, exog_cols]

    forecast = result.forecast(steps=len(future_dates), exog=X_future).to_frame(name='predicted_mean')

    if diff:
        last_val = train_data_cpr[y_col].loc['2024-12-01']
        forecast['predicted_mean'] = forecast['predicted_mean'].cumsum() + last_val

    # --- Inyectar predicción en dataframe completo ---
    col_forecast = f'fcp{numero}f'
    df_forecast = df_forecast.copy()
    df_forecast[col_forecast] = df_forecast[y_col]
    df_forecast.loc[forecast.index, col_forecast] = forecast['predicted_mean']

    # --- Filtrar y aplicar HP ---
    serie_completa = df_forecast[col_forecast].loc['2006-03-01':'2025-12-01']
    _, hp_cycle = sm.tsa.filters.hpfilter(serie_completa, lamb=1)

    # Nueva salida: la serie sin HP
    serie_sin_hp = serie_completa.rename(f'FCP{numero}F')

    return hp_cycle.rename(f'FCP{numero}F_H'), serie_sin_hp, df_forecast


forecast_data_cpr_mod = forecast_data_cpr.copy()

FCP1F_H, FCP1F, forecast_data_cpr_mod = run_model_cpr(
    forecast_data_cpr_mod, 1, 'fcp1', ['find1f', 'fcon1f'],
    order=([1, 0, 1], 0, 1), seasonal_order=(1, 0, 0, 12), trend='n', start_date='2006-03-01'
)

FCP2F_H, FCP2F, forecast_data_cpr_mod = run_model_cpr(
    forecast_data_cpr_mod, 2, 'fcp2', None,
    order=([1, 1], 0, 0), seasonal_order=(2, 0, 0, 12), trend='n',
    start_date='2006-04-01', diff=True
)

FCP3F_H, FCP3F, forecast_data_cpr_mod = run_model_cpr(
    forecast_data_cpr_mod, 3, 'fcp3', ['find2f'],
    order=(0, 0, 3), seasonal_order=(0, 0, 1, 12), trend='n',
    start_date='2006-04-01', diff=True, diff_exog=True
)

FCP4F_H, FCP4F, forecast_data_cpr_mod = run_model_cpr(
    forecast_data_cpr_mod, 4, 'fcp4', None,
    order=([0, 1], 0, 3), seasonal_order=(0, 0, 1, 12), trend='n',
    start_date='2006-04-01', diff=True
)

FCP5F_H, FCP5F, forecast_data_cpr_mod = run_model_cpr(
    forecast_data_cpr_mod, 5, 'fcp5', ['f202005'],
    order=(2, 0, 0), seasonal_order=(1, 0, 0, 12), trend='n',
    start_date='2006-03-01'
)

FCP6F_H, FCP6F, forecast_data_cpr_mod = run_model_cpr(
    forecast_data_cpr_mod, 6, 'fcp6', ['find3f', 'fcon3f'],
    order=(0, 0, 1), seasonal_order=(0, 0, 1, 12), trend='n',
    start_date='2006-04-01', diff=True, diff_exog=True
)

FCP7F_H, FCP7F, forecast_data_cpr_mod = run_model_cpr(
    forecast_data_cpr_mod, 7, 'fcp7', ['f202005'],
    order=([1, 0, 1], 0, 0), seasonal_order=(1, 0, 0, 12), trend='n',
    start_date='2006-04-01', diff=True, diff_exog=True
)
output_path_cpr = r'C:\Users\lucas\MAF_ESP\Forecast_CPR_HP.csv'
hp_series_cpr = pd.concat([
    FCP1F_H, FCP2F_H, FCP3F_H, FCP4F_H, FCP5F_H, FCP6F_H, FCP7F_H
], axis=1)
hp_series_cpr.to_csv(output_path_cpr)

# Guardar series sin filtro HP
s_series_cpr = pd.concat([FCP1F, FCP2F, FCP3F, FCP4F, FCP5F, FCP6F, FCP7F], axis=1)
s_series_cpr.to_csv(r'C:\Users\lucas\High-Frequency_GDP_Forecasting\src\UNION\Exogenas_sin_HP\CPR_sin_HP.csv')

# ESTA PERFECTO

# ---------- CARGA DE DATOS ----------
variables_arima_ser_full = pd.read_csv(
    r'C:\Users\lucas\High-Frequency_GDP_Forecasting\src\UNION\FactoresPCA\factores_SER.txt',
    delimiter='\t', parse_dates=['_date_'], index_col='_date_'
).asfreq('MS')

# Cargar ambas columnas desde archivo
col_fexp1f = pd.read_csv(r'C:\Users\lucas\High-Frequency_GDP_Forecasting\src\UNION\Exogenas\fexp1f.csv', index_col=0, parse_dates=True)
col_fcp1f = pd.read_csv(r'C:\Users\lucas\High-Frequency_GDP_Forecasting\src\UNION\Exogenas\fcp1f.csv', index_col=0, parse_dates=True)
col_fcp3f = pd.read_csv(r'C:\Users\lucas\High-Frequency_GDP_Forecasting\src\UNION\Exogenas\fcp3f.csv', index_col=0, parse_dates=True)
col_fcp4f = pd.read_csv(r'C:\Users\lucas\High-Frequency_GDP_Forecasting\src\UNION\Exogenas\fcp4f.csv', index_col=0, parse_dates=True)
col_fexp3f = pd.read_csv(r'C:\Users\lucas\High-Frequency_GDP_Forecasting\src\UNION\Exogenas\fexp3f.csv', index_col=0, parse_dates=True)
col_fexp4f = pd.read_csv(r'C:\Users\lucas\High-Frequency_GDP_Forecasting\src\UNION\Exogenas\fexp4f.csv', index_col=0, parse_dates=True)
col_f202005 = pd.read_csv(r'C:\Users\lucas\High-Frequency_GDP_Forecasting\src\UNION\Exogenas\f202005.csv', index_col=0, parse_dates=True)
col_f202006 = pd.read_csv(r'C:\Users\lucas\High-Frequency_GDP_Forecasting\src\UNION\Exogenas\f202006.csv', index_col=0, parse_dates=True)
# Cargar DataFrame base
variables_arima_ser_full = pd.read_csv(r'C:\Users\lucas\High-Frequency_GDP_Forecasting\src\UNION\FactoresPCA\factores_SER.txt',delimiter='\t', parse_dates=['_date_'], index_col='_date_').asfreq('MS')
# Unir ambas columnas al DataFrame original
variables_arima_ser_full = variables_arima_ser_full.join([col_fexp1f, col_fcp1f, col_fcp3f, col_fcp4f, col_fexp3f, col_fexp4f, col_f202005, col_f202006])

train_data_ser = variables_arima_ser_full.loc['1996-01-01':'2024-12-01']
forecast_data_ser = variables_arima_ser_full.loc['1996-01-01':'2025-12-01']


def run_model_ser(df_forecast, numero, y_col, exog_cols=None, order=(1, 0, 0),
                  seasonal_order=(0, 0, 0, 0), trend='n', start_date='2003-04-01',
                  diff=False, diff_exog_cols=None, verbose=True):
    # --- Preparar Y ---
    if diff:
        y_full = train_data_ser[y_col].diff()
        y = y_full.loc[start_date:]
    else:
        y = train_data_ser[y_col].loc[start_date:]
    y = y.dropna()

    # --- Preparar X ---
    X = None
    if exog_cols:
        X_raw = train_data_ser[exog_cols].copy()

        if diff_exog_cols:
            for col in diff_exog_cols:
                if col in X_raw.columns:
                    X_raw[col] = X_raw[col].diff()

        X = X_raw.loc[start_date:]
        X = X.loc[y.index]
        X = X[~X.isin([float('inf'), float('-inf')]).any(axis=1) & X.notna().all(axis=1)]
        y = y.loc[X.index]

    # --- Ajuste ARIMA ---
    model = ARIMA(y, exog=X, order=order, seasonal_order=seasonal_order, trend=trend)
    result = model.fit()

    # if verbose:
    #     print(f"\n--- Modelo fser{numero} ---")
    #     print(result.summary())

    # --- Forecast ---
    steps = 12
    future_dates = pd.date_range(start='2025-01-01', periods=steps, freq='MS')

    X_future = None
    if exog_cols:
        extended_dates = pd.date_range(start='2024-12-01', periods=steps + 1, freq='MS')
        X_future_raw = df_forecast.loc[extended_dates, exog_cols].copy()

        if diff_exog_cols:
            for col in diff_exog_cols:
                if col in X_future_raw.columns:
                    X_future_raw[col] = X_future_raw[col].diff()

            X_future = X_future_raw.dropna()
        else:
            X_future = X_future_raw.loc[future_dates]

    forecast = result.forecast(steps=steps, exog=X_future).to_frame(name='predicted_mean')

    if diff:
        last_val = train_data_ser[y_col].loc['2024-12-01']
        forecast['predicted_mean'] = forecast['predicted_mean'].cumsum() + last_val

    col_forecast = f'fser{numero}f'
    df_forecast = df_forecast.copy()
    df_forecast[col_forecast] = df_forecast[y_col]
    df_forecast.loc[forecast.index, col_forecast] = forecast['predicted_mean']

    serie_completa = df_forecast[col_forecast].loc['2006-03-01':'2025-12-01']
    _, hp_cycle = sm.tsa.filters.hpfilter(serie_completa, lamb=1)

    # Nueva salida: la serie sin HP
    serie_sin_hp = serie_completa.rename(f'FSER{numero}F')

    return hp_cycle.rename(f'FSER{numero}F_H'), serie_sin_hp, df_forecast


forecast_data_ser_mod = forecast_data_ser.copy()

FSER1F_H, FSER1F, forecast_data_ser_mod = run_model_ser(
    forecast_data_ser_mod, 1, 'fser1', ['fexp1f', 'fcp1f'],
    order=(2, 0, [0, 0, 1]), seasonal_order=(0, 0, 1, 12),
    trend='n', start_date='2006-03-01'
)

# Ecuación 2
FSER2F_H, FSER2F, forecast_data_ser_mod = run_model_ser(
    forecast_data_ser_mod, 2, 'fser2',
    exog_cols=['fcp3f', 'fexp1f', 'f202006'],
    diff=True,
    diff_exog_cols=['fcp3f', 'fexp1f'],
    order=(1, 0, [0, 0, 1]), seasonal_order=(0, 0, 1, 12), trend='n',
    start_date='2006-04-01'
)

# Ecuación 3
FSER3F_H, FSER3F, forecast_data_ser_mod = run_model_ser(
    forecast_data_ser_mod, 3, 'fser3',
    exog_cols=['fcp4f', 'fexp3f', 'f202006'],
    diff=True,
    diff_exog_cols=['fcp4f', 'fexp3f'],
    order=(0, 0, 1), seasonal_order=(0, 0, 1, 12), trend='n',
    start_date='2006-04-01'
)

# Ecuación 4
FSER4F_H, FSER4F, forecast_data_ser_mod = run_model_ser(
    forecast_data_ser_mod, 4, 'fser4',
    exog_cols=['fcp3f', 'f202006', 'f202005'],
    diff=True,
    diff_exog_cols=['fcp3f'],
    order=([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 0, [1, 0, 1]),
    seasonal_order=(0, 0, 0, 0), trend='n',
    start_date='2006-04-01'
)

# Ecuación 5
FSER5F_H, FSER5F, forecast_data_ser_mod = run_model_ser(
    forecast_data_ser_mod, 5, 'fser5',
    exog_cols=['fcp1f', 'f202005'],
    diff=True,
    diff_exog_cols=['fcp1f'],
    order=(1, 0, [0, 1]), seasonal_order=(0, 0, 1, 12), trend='n',
    start_date='2006-04-01'
)

FSER6F_H, FSER6F, forecast_data_ser_mod = run_model_ser(
    forecast_data_ser_mod, 6, 'fser6',
    exog_cols=['fexp4f'],
    diff=True,
    diff_exog_cols=['fexp4f'],
    order=(1, 0, [0, 0, 1]), seasonal_order=(0, 0, 1, 12),
    trend='n', start_date='2006-04-01'
)
# ---------- GUARDAR SERIES HP ----------
output_path_ser = r'C:\Users\lucas\MAF_ESP\Forecast_SER_HP.csv'
hp_series_ser = pd.concat([
    FSER1F_H, FSER2F_H, FSER3F_H, FSER4F_H, FSER5F_H, FSER6F_H
], axis=1)
hp_series_ser.to_csv(output_path_ser)

# Guardar series sin filtro HP
s_series_ser = pd.concat([FSER1F, FSER2F, FSER3F, FSER4F, FSER5F, FSER6F], axis=1)
s_series_ser.to_csv(r'C:\Users\lucas\High-Frequency_GDP_Forecasting\src\UNION\Exogenas_sin_HP\SER_sin_HP.csv')

# ESTA PERFECTO

# ---------- CARGA DE DATOS ----------
variables_arima_con_full = pd.read_csv(
    r'C:\Users\lucas\High-Frequency_GDP_Forecasting\src\UNION\FactoresPCA\factores_CON.txt',
    delimiter='\t', parse_dates=['_date_'], index_col='_date_'
).asfreq('MS')

# Cargar ambas columnas desde archivo
col_finv1f = pd.read_csv(r'C:\Users\lucas\High-Frequency_GDP_Forecasting\src\UNION\Exogenas\finv1f.csv', index_col=0, parse_dates=True)
col_finv2f = pd.read_csv(r'C:\Users\lucas\High-Frequency_GDP_Forecasting\src\UNION\Exogenas\finv2f.csv', index_col=0, parse_dates=True)
col_finv3f = pd.read_csv(r'C:\Users\lucas\High-Frequency_GDP_Forecasting\src\UNION\Exogenas\finv3f.csv', index_col=0, parse_dates=True)
# Cargar DataFrame base
variables_arima_con_full = pd.read_csv(r'C:\Users\lucas\High-Frequency_GDP_Forecasting\src\UNION\FactoresPCA\factores_CON.txt',delimiter='\t', parse_dates=['_date_'], index_col='_date_').asfreq('MS')
# Unir ambas columnas al DataFrame original
variables_arima_con_full = variables_arima_con_full.join([col_finv1f, col_finv2f, col_finv3f])

train_data_con = variables_arima_con_full.loc['1996-01-01':'2024-12-01']
forecast_data_con = variables_arima_con_full.loc['1996-01-01':'2025-12-01']


def run_model_con(df_forecast, numero, y_col, exog_cols=None, order=(1, 0, 0),
                  seasonal_order=(0, 0, 0, 0), trend='n', start_date='2003-04-01',
                  diff=False, diff_exog_cols=None, verbose=True, use_sarimax=False):
    if diff:
        y_full = train_data_con[y_col].diff()
        y = y_full.loc[start_date:]
    else:
        y = train_data_con[y_col].loc[start_date:]
    y = y.dropna()

    X = None
    if exog_cols:
        X_raw = train_data_con[exog_cols].copy()
        if diff_exog_cols:
            for col in diff_exog_cols:
                if col in X_raw.columns:
                    X_raw[col] = X_raw[col].diff()

        X = X_raw.loc[start_date:]
        X = X.loc[y.index]
        X = X[~X.isin([float('inf'), float('-inf')]).any(axis=1) & X.notna().all(axis=1)]
        y = y.loc[X.index]

    # Elegir clase de modelo
    ModelClass = sm.tsa.SARIMAX if use_sarimax else ARIMA
    model = ModelClass(y, exog=X, order=order, seasonal_order=seasonal_order, trend=trend)

    # Fit según tipo de modelo
    if use_sarimax:
        result = model.fit(method='bfgs', maxiter=1000, disp=False)
    else:
        result = model.fit()

    # if verbose:
    #     print(f"\n--- Modelo fcon{numero} ---")
    #     print(result.summary())

    steps = 12
    future_dates = pd.date_range(start='2025-01-01', periods=steps, freq='MS')

    X_future = None
    if exog_cols:
        extended_dates = pd.date_range(start='2024-12-01', periods=steps + 1, freq='MS')
        X_future_raw = df_forecast.loc[extended_dates, exog_cols].copy()
        if diff_exog_cols:
            for col in diff_exog_cols:
                if col in X_future_raw.columns:
                    X_future_raw[col] = X_future_raw[col].diff()
            X_future = X_future_raw.dropna()
        else:
            X_future = X_future_raw.loc[future_dates]

    forecast = result.forecast(steps=steps, exog=X_future).to_frame(name='predicted_mean')

    if diff:
        last_val = train_data_con[y_col].loc['2024-12-01']
        forecast['predicted_mean'] = forecast['predicted_mean'].cumsum() + last_val

    col_forecast = f'fcon{numero}f'
    df_forecast = df_forecast.copy()
    df_forecast[col_forecast] = df_forecast[y_col]
    df_forecast.loc[forecast.index, col_forecast] = forecast['predicted_mean']

    serie_completa = df_forecast[col_forecast].loc['2003-03-01':'2025-12-01']
    _, hp_cycle = sm.tsa.filters.hpfilter(serie_completa, lamb=1)

    # Nueva salida: la serie sin HP
    serie_sin_hp = serie_completa.rename(f'FCON{numero}F')

    return hp_cycle.rename(f'FCON{numero}F_H'), serie_sin_hp, df_forecast


# Carga de datos
variables_arima_con_full = pd.read_csv(
    r'C:\Users\lucas\High-Frequency_GDP_Forecasting\src\UNION\FactoresPCA\factores_CON.txt',
    delimiter='\t', parse_dates=['_date_'], index_col='_date_'
).asfreq('MS')

# Cargar ambas columnas desde archivo
col_finv1f = pd.read_csv(r'C:\Users\lucas\High-Frequency_GDP_Forecasting\src\UNION\Exogenas\finv1f.csv', index_col=0, parse_dates=True)
col_finv2f = pd.read_csv(r'C:\Users\lucas\High-Frequency_GDP_Forecasting\src\UNION\Exogenas\finv2f.csv', index_col=0, parse_dates=True)
col_finv3f = pd.read_csv(r'C:\Users\lucas\High-Frequency_GDP_Forecasting\src\UNION\Exogenas\finv3f.csv', index_col=0, parse_dates=True)
col_inv2f = pd.read_csv(r'C:\Users\lucas\High-Frequency_GDP_Forecasting\src\UNION\Exogenas\inv2f.csv', index_col=0, parse_dates=True)

# Cargar DataFrame base
variables_arima_con_full = pd.read_csv(r'C:\Users\lucas\High-Frequency_GDP_Forecasting\src\UNION\FactoresPCA\factores_CON.txt',delimiter='\t', parse_dates=['_date_'], index_col='_date_').asfreq('MS')
# Unir ambas columnas al DataFrame original
variables_arima_con_full = variables_arima_con_full.join([col_finv1f, col_finv2f, col_finv3f, col_inv2f])


train_data_con = variables_arima_con_full.loc['1996-01-01':'2024-12-01']
forecast_data_con = variables_arima_con_full.loc['1996-01-01':'2025-12-01']
forecast_data_con_mod = forecast_data_con.copy()

# Ecuación 1
FCON1F_H, FCON1F, forecast_data_con_mod = run_model_con(
    forecast_data_con_mod, 1, 'fcon1',
    exog_cols=['finv1f'], order=(1, 0, 3),
    seasonal_order=(0, 0, 0, 0), trend='n',
    start_date='2003-03-01'
)

# Ecuación 2
FCON2F_H, FCON2F, forecast_data_con_mod = run_model_con(
    forecast_data_con_mod, 2, 'fcon2',
    exog_cols=['finv3f'], order=(1, 0, [1, 0, 1]),
    seasonal_order=(0, 0, 0, 0), trend='n',
    start_date='2003-04-01', diff=True,
    diff_exog_cols=['finv3f']
)

# Ecuación 3 (usar SARIMAX)
FCON3F_H, FCON3F, forecast_data_con_mod = run_model_con(
    forecast_data_con_mod, 3, 'fcon3',
    exog_cols=['inv2f'], order=(1, 0, [0, 0, 1]),
    seasonal_order=(0, 0, 1, 12), trend='n',
    start_date='2002-04-01', diff=True,
    diff_exog_cols=['inv2f'], use_sarimax=True
)

# Ecuación 4
FCON4F_H, FCON4F, forecast_data_con_mod = run_model_con(
    forecast_data_con_mod, 4, 'fcon4',
    exog_cols=['finv1f'], order=([1, 0, 1, 1], 0, 0),
    seasonal_order=(0, 0, 1, 12), trend='n',
    start_date='2003-04-01', diff=True,
    diff_exog_cols=['finv1f']
)

# Ecuación 5
FCON5F_H, FCON5F, forecast_data_con_mod = run_model_con(
    forecast_data_con_mod, 5, 'fcon5',
    exog_cols=['finv1f'], order=(1, 0, [0, 0, 1, 1]),
    seasonal_order=(0, 0, 1, 12), trend='n',
    start_date='2003-04-01', diff=True,
    diff_exog_cols=['finv1f']
)
# Guardar series con filtro HP
output_path_con = r'C:\Users\lucas\MAF_ESP\Forecast_CON_HP.csv'
hp_series_con = pd.concat([
    FCON1F_H, FCON2F_H, FCON3F_H, FCON4F_H, FCON5F_H
], axis=1)
hp_series_con.to_csv(output_path_con)

# Guardar series sin filtro HP
s_series_con = pd.concat([FCON1F, FCON2F, FCON3F, FCON4F, FCON5F], axis=1)
s_series_con.to_csv(r'C:\Users\lucas\High-Frequency_GDP_Forecasting\src\UNION\Exogenas_sin_HP\CON_sin_HP.csv')
# ESTA PERFECTO excepto la ecuacion 1

# ---------- CARGA DE DATOS ----------
variables_arima_imp_full = pd.read_csv(
    r'C:\Users\lucas\High-Frequency_GDP_Forecasting\src\UNION\FactoresPCA\factores_IMP.txt',
    delimiter='\t', parse_dates=['_date_'], index_col='_date_'
).asfreq('MS')

# Cargar ambas columnas desde archivo
col_fexp1f = pd.read_csv(r'C:\Users\lucas\High-Frequency_GDP_Forecasting\src\UNION\Exogenas\fexp1f.csv', index_col=0, parse_dates=True)
col_fcp1f = pd.read_csv(r'C:\Users\lucas\High-Frequency_GDP_Forecasting\src\UNION\Exogenas\fcp1f.csv', index_col=0, parse_dates=True)
col_fcp4f = pd.read_csv(r'C:\Users\lucas\High-Frequency_GDP_Forecasting\src\UNION\Exogenas\fcp4f.csv', index_col=0, parse_dates=True)
col_finv3f = pd.read_csv(r'C:\Users\lucas\High-Frequency_GDP_Forecasting\src\UNION\Exogenas\finv3f.csv', index_col=0, parse_dates=True)

# Cargar DataFrame base
variables_arima_imp_full = pd.read_csv(r'C:\Users\lucas\High-Frequency_GDP_Forecasting\src\UNION\FactoresPCA\factores_IMP.txt',delimiter='\t', parse_dates=['_date_'], index_col='_date_').asfreq('MS')
# Unir ambas columnas al DataFrame original
variables_arima_imp_full = variables_arima_imp_full.join([col_fexp1f, col_fcp1f, col_fcp4f, col_finv3f])


train_data_imp = variables_arima_imp_full.loc['1996-01-01':'2024-12-01']
forecast_data_imp = variables_arima_imp_full.loc['1996-01-01':'2025-12-01']


def run_model_imp(df_forecast, numero, y_col, exog_cols=None, order=(1, 0, 0),
                  seasonal_order=(0, 0, 0, 0), trend='n', start_date='2003-04-01',
                  diff=False, diff_exog_cols=None, verbose=True, use_sarimax=False):
    if diff:
        y_full = train_data_imp[y_col].diff()
        y = y_full.loc[start_date:]
    else:
        y = train_data_imp[y_col].loc[start_date:]
    y = y.dropna()

    X = None
    if exog_cols:
        X_raw = train_data_imp[exog_cols].copy()
        if diff_exog_cols:
            for col in diff_exog_cols:
                if col in X_raw.columns:
                    X_raw[col] = X_raw[col].diff()

        X = X_raw.loc[start_date:]
        X = X.loc[y.index]
        X = X[~X.isin([float('inf'), float('-inf')]).any(axis=1) & X.notna().all(axis=1)]
        y = y.loc[X.index]

    ModelClass = sm.tsa.SARIMAX if use_sarimax else ARIMA
    model = ModelClass(y, exog=X, order=order, seasonal_order=seasonal_order, trend=trend)
    result = model.fit(method='lbfgs', maxiter=1000, disp=False) if use_sarimax else model.fit()

    # if verbose:
    #     print(f"\n--- Modelo fimp{numero} ---")
    #     print(result.summary())

    steps = 14
    future_dates = pd.date_range(start='2024-10-01', periods=steps,
                                 freq='MS')  # O '2025-01-01' si tienes datos completos

    X_future = None
    if exog_cols:
        extended_dates = pd.date_range(start='2024-10-01', periods=steps + 1, freq='MS')
        X_future_raw = df_forecast.loc[extended_dates, exog_cols].copy()
        if diff_exog_cols:
            for col in diff_exog_cols:
                if col in X_future_raw.columns:
                    X_future_raw[col] = X_future_raw[col].diff()
            X_future = X_future_raw.dropna()
        else:
            X_future = X_future_raw.loc[future_dates]

    forecast = result.forecast(steps=steps, exog=X_future).to_frame(name='predicted_mean')

    if diff:
        last_val = train_data_imp[y_col].loc['2024-10-01']
        forecast['predicted_mean'] = forecast['predicted_mean'].cumsum() + last_val

    col_forecast = f'fimp{numero}f'
    df_forecast = df_forecast.copy()
    df_forecast[col_forecast] = df_forecast[y_col]
    df_forecast.loc[forecast.index, col_forecast] = forecast['predicted_mean']

    serie_completa = df_forecast[col_forecast].loc['1997-03-01':'2025-12-01']
    _, hp_cycle = sm.tsa.filters.hpfilter(serie_completa, lamb=1)

    # Nueva salida: la serie sin HP
    serie_sin_hp = serie_completa.rename(f'FIMP{numero}F')

    return hp_cycle.rename(f'FIMP{numero}F_H'), serie_sin_hp, df_forecast


# Cargar y preparar
variables_arima_imp_full = pd.read_csv(
    r'C:\Users\lucas\High-Frequency_GDP_Forecasting\src\UNION\FactoresPCA\factores_IMP.txt',
    delimiter='\t', parse_dates=['_date_'], index_col='_date_'
).asfreq('MS')

# Cargar ambas columnas desde archivo
col_fexp1f = pd.read_csv(r'C:\Users\lucas\High-Frequency_GDP_Forecasting\src\UNION\Exogenas\fexp1f.csv', index_col=0, parse_dates=True)
col_fcp1f = pd.read_csv(r'C:\Users\lucas\High-Frequency_GDP_Forecasting\src\UNION\Exogenas\fcp1f.csv', index_col=0, parse_dates=True)
col_fcp4f = pd.read_csv(r'C:\Users\lucas\High-Frequency_GDP_Forecasting\src\UNION\Exogenas\fcp4f.csv', index_col=0, parse_dates=True)
col_finv3f = pd.read_csv(r'C:\Users\lucas\High-Frequency_GDP_Forecasting\src\UNION\Exogenas\finv3f.csv', index_col=0, parse_dates=True)

# Cargar DataFrame base
variables_arima_imp_full = pd.read_csv(r'C:\Users\lucas\High-Frequency_GDP_Forecasting\src\UNION\FactoresPCA\factores_IMP.txt',delimiter='\t', parse_dates=['_date_'], index_col='_date_').asfreq('MS')
# Unir ambas columnas al DataFrame original
variables_arima_imp_full = variables_arima_imp_full.join([col_fexp1f, col_fcp1f, col_fcp4f, col_finv3f])

train_data_imp = variables_arima_imp_full.loc['1996-01-01':'2024-10-01']
forecast_data_imp = variables_arima_imp_full.loc['1996-01-01':'2025-12-01']
forecast_data_imp_mod = forecast_data_imp.copy()

# Ecuación 1
FIMP1F_H, FIMP1F, forecast_data_imp_mod = run_model_imp(
    forecast_data_imp_mod, 1, 'fimp1',
    exog_cols=['fexp1f', 'fcp1f'],
    order=(2, 0, [0, 0, 1]), seasonal_order=(0, 0, 1, 12),
    trend='n', start_date='2006-03-01', use_sarimax=True
)

# Ecuación 2
FIMP2F_H, FIMP2F, forecast_data_imp_mod = run_model_imp(
    forecast_data_imp_mod, 2, 'fimp2',
    exog_cols=['fcp4f', 'fexp1f'],
    order=(1, 0, [0] * 11 + [1]), seasonal_order=(0, 0, 0, 0),
    trend='n', start_date='2006-04-01',
    diff=True, diff_exog_cols=['fcp4f', 'fexp1f']
)

# Ecuación 3 (sin exógenas)
FIMP3F_H, FIMP3F,  forecast_data_imp_mod = run_model_imp(
    forecast_data_imp_mod, 3, 'fimp3',
    exog_cols=None,
    order=(2, 0, [0, 0, 0, 1]), seasonal_order=(0, 0, 1, 12),
    trend='n', start_date='1997-04-01',
    diff=True
)

# Ecuación 4
FIMP4F_H, FIMP4F,  forecast_data_imp_mod = run_model_imp(
    forecast_data_imp_mod, 4, 'fimp4',
    exog_cols=['finv3f'],
    order=([0, 1], 0, [0] * 11 + [1]), seasonal_order=(0, 0, 0, 0),
    trend='n', start_date='2003-04-01',
    diff=True, diff_exog_cols=['finv3f']
)
# Guardar series CON filtro HP
output_path_imp = r'C:\Users\lucas\MAF_ESP\Forecast_IMP_HP.csv'
hp_series_imp = pd.concat([
    FIMP1F_H, FIMP2F_H, FIMP3F_H, FIMP4F_H
], axis=1)
hp_series_imp.to_csv(output_path_imp)

# Guardar series sin filtro HP
s_series_imp = pd.concat([FIMP1F, FIMP2F, FIMP3F, FIMP4F], axis=1)
s_series_imp.to_csv(r'C:\Users\lucas\High-Frequency_GDP_Forecasting\src\UNION\Exogenas_sin_HP\IMP_sin_HP.csv')

# ESTA PERFECTO

# ---------- CARGA DE DATOS ----------
variables_arima_impu_full = pd.read_csv(
    r'C:\Users\lucas\High-Frequency_GDP_Forecasting\src\UNION\FactoresPCA\factores_IMPU.txt',
    delimiter='\t', parse_dates=['_date_'], index_col='_date_'
).asfreq('MS')

# Cargar ambas columnas desde archivo
col_f200301 = pd.read_csv(r'C:\Users\lucas\High-Frequency_GDP_Forecasting\src\UNION\Exogenas\f200301.csv', index_col=0, parse_dates=True)
col_f200302 = pd.read_csv(r'C:\Users\lucas\High-Frequency_GDP_Forecasting\src\UNION\Exogenas\f200302.csv', index_col=0, parse_dates=True)
# Cargar DataFrame base
variables_arima_impu_full = pd.read_csv(r'C:\Users\lucas\High-Frequency_GDP_Forecasting\src\UNION\FactoresPCA\factores_IMPU.txt',delimiter='\t', parse_dates=['_date_'], index_col='_date_').asfreq('MS')
# Unir ambas columnas al DataFrame original
variables_arima_impu_full = variables_arima_impu_full.join([col_f200301, col_f200302])

train_data_impu = variables_arima_impu_full.loc['1996-01-01':'2024-12-01']
forecast_data_impu = variables_arima_impu_full.loc['1996-01-01':'2025-12-01']


def run_model_impu(df_forecast, numero, y_col, exog_cols=None, order=(1, 0, 0),
                   seasonal_order=(0, 0, 0, 0), trend='n', start_date='2003-04-01',
                   diff=False, diff_exog_cols=None, verbose=True, use_sarimax=False):
    if diff:
        y_full = train_data_impu[y_col].diff()
        y = y_full.loc[start_date:]
    else:
        y = train_data_impu[y_col].loc[start_date:]
    y = y.dropna()

    X = None
    if exog_cols:
        X_raw = train_data_impu[exog_cols].copy()
        if diff_exog_cols:
            for col in diff_exog_cols:
                if col in X_raw.columns:
                    X_raw[col] = X_raw[col].diff()

        X = X_raw.loc[start_date:]
        X = X.loc[y.index]
        X = X[~X.isin([float('inf'), float('-inf')]).any(axis=1) & X.notna().all(axis=1)]
        y = y.loc[X.index]

    ModelClass = sm.tsa.SARIMAX if use_sarimax else sm.tsa.ARIMA
    model = ModelClass(y, exog=X, order=order, seasonal_order=seasonal_order, trend=trend)
    result = model.fit(method='bfgs', maxiter=1000, disp=False) if use_sarimax else model.fit()

    # if verbose:
    #     print(f"\n--- Modelo fimpu{numero} ---")
    #     print(result.summary())

    steps = 14
    future_dates = pd.date_range(start='2024-11-01', periods=steps, freq='MS')

    X_future = None
    if exog_cols:
        extended_dates = pd.date_range(start='2024-10-01', periods=steps + 1, freq='MS')
        X_future_raw = df_forecast.loc[extended_dates, exog_cols].copy()

        if diff_exog_cols:
            for col in diff_exog_cols:
                if col in X_future_raw.columns:
                    X_future_raw[col] = X_future_raw[col].diff()
            X_future = X_future_raw.dropna()
        else:
            X_future = X_future_raw.loc[future_dates]

    forecast = result.forecast(steps=steps, exog=X_future).to_frame(name='predicted_mean')

    if diff:
        last_val = train_data_impu[y_col].loc['2024-10-01']
        forecast['predicted_mean'] = forecast['predicted_mean'].cumsum() + last_val

    col_forecast = f'fimpu{numero}f'
    df_forecast = df_forecast.copy()
    df_forecast[col_forecast] = df_forecast[y_col]
    df_forecast.loc[forecast.index, col_forecast] = forecast['predicted_mean']

    serie_completa = df_forecast[col_forecast].loc['1996-03-01':'2025-12-01']
    _, hp_cycle = sm.tsa.filters.hpfilter(serie_completa, lamb=1)

    # Nueva salida: la serie sin HP
    serie_sin_hp = serie_completa.rename(f'FIMPU{numero}F')

    return hp_cycle.rename(f'FIMPU{numero}F_H'),serie_sin_hp, df_forecast


# Cargar datos
variables_arima_impu_full = pd.read_csv(
    r'C:\Users\lucas\High-Frequency_GDP_Forecasting\src\UNION\FactoresPCA\factores_IMPU.txt',
    delimiter='\t', parse_dates=['_date_'], index_col='_date_'
).asfreq('MS')

# Cargar ambas columnas desde archivo
col_f200301 = pd.read_csv(r'C:\Users\lucas\High-Frequency_GDP_Forecasting\src\UNION\Exogenas\f200301.csv', index_col=0, parse_dates=True)
col_f200302 = pd.read_csv(r'C:\Users\lucas\High-Frequency_GDP_Forecasting\src\UNION\Exogenas\f200302.csv', index_col=0, parse_dates=True)
# Cargar DataFrame base
variables_arima_impu_full = pd.read_csv(r'C:\Users\lucas\High-Frequency_GDP_Forecasting\src\UNION\FactoresPCA\factores_IMPU.txt',delimiter='\t', parse_dates=['_date_'], index_col='_date_').asfreq('MS')
# Unir ambas columnas al DataFrame original
variables_arima_impu_full = variables_arima_impu_full.join([col_f200301, col_f200302])

train_data_impu = variables_arima_impu_full.loc['1996-01-01':'2024-10-01']
forecast_data_impu = variables_arima_impu_full.loc['1996-01-01':'2025-12-01']
forecast_data_impu_mod = forecast_data_impu.copy()

# Ecuación 1 (usar SARIMAX con lbfgs)
FIMPU1F_H, FIMPU1F, forecast_data_impu_mod = run_model_impu(
    forecast_data_impu_mod, 1, 'fimpu1',
    order=(2, 0, [1, 0, 1]), seasonal_order=(1, 0, 0, 12),
    trend='n', start_date='1996-03-01',
    use_sarimax=False
)

# Ecuación 2
FIMPU2F_H, FIMPU2F, forecast_data_impu_mod = run_model_impu(
    forecast_data_impu_mod, 2, 'fimpu2',
    order=(1, 0, [0, 0, 1]), seasonal_order=(0, 0, 1, 12),
    trend='n', start_date='1996-04-01',
    diff=True
)

# Ecuación 3
FIMPU3F_H, FIMPU3F, forecast_data_impu_mod = run_model_impu(
    forecast_data_impu_mod, 3, 'fimpu3',
    exog_cols=['f200301'],
    order=(3, 0, 1), seasonal_order=(1, 0, 1, 12),
    trend='n', start_date='1996-03-01',
    use_sarimax=True
)

# Ecuación 4
FIMPU4F_H, FIMPU4F, forecast_data_impu_mod = run_model_impu(
    forecast_data_impu_mod, 4, 'fimpu4',
    exog_cols=['f200302'],
    order=(1, 0, [1, 0, 1]), seasonal_order=(0, 0, 0, 0),
    trend='n', start_date='1996-03-01'
)

# Guardar series con filtro HP
output_path_impu = r'C:\Users\lucas\MAF_ESP\Forecast_IMPU_HP.csv'
hp_series_impu = pd.concat([
    FIMPU1F_H, FIMPU2F_H, FIMPU3F_H, FIMPU4F_H
], axis=1)
hp_series_impu.to_csv(output_path_impu)

# Guardar series sin filtro HP
s_series_impu = pd.concat([FIMPU1F, FIMPU2F, FIMPU3F, FIMPU4F], axis=1)
s_series_impu.to_csv(r'C:\Users\lucas\High-Frequency_GDP_Forecasting\src\UNION\Exogenas_sin_HP\IMPU_sin_HP.csv')

# ESTA PERFECTO excepto la ecuacion 1

# ---------- CARGA DE DATOS ----------
variables_arima_exp_full = pd.read_csv(
    r'C:\Users\lucas\High-Frequency_GDP_Forecasting\src\UNION\FactoresPCA\factores_EXP.txt',
    delimiter='\t', parse_dates=['_date_'], index_col='_date_'
).asfreq('MS')

# Cargar ambas columnas desde archivo
col_pibuem = pd.read_csv(r'C:\Users\lucas\High-Frequency_GDP_Forecasting\src\UNION\Exogenas\pibuem.csv', index_col=0, parse_dates=True)
col_fvirus = pd.read_csv(r'C:\Users\lucas\High-Frequency_GDP_Forecasting\src\UNION\Exogenas\fvirus.csv', index_col=0, parse_dates=True)
col_fcexpor = pd.read_csv(r'C:\Users\lucas\High-Frequency_GDP_Forecasting\src\UNION\Exogenas\fcexpor.csv', index_col=0, parse_dates=True)
col_f202004 = pd.read_csv(r'C:\Users\lucas\High-Frequency_GDP_Forecasting\src\UNION\Exogenas\f202004.csv', index_col=0, parse_dates=True)
col_f202005 = pd.read_csv(r'C:\Users\lucas\High-Frequency_GDP_Forecasting\src\UNION\Exogenas\f202005.csv', index_col=0, parse_dates=True)
col_f202006 = pd.read_csv(r'C:\Users\lucas\High-Frequency_GDP_Forecasting\src\UNION\Exogenas\f202006.csv', index_col=0, parse_dates=True)
# Cargar DataFrame base
variables_arima_exp_full = pd.read_csv(r'C:\Users\lucas\High-Frequency_GDP_Forecasting\src\UNION\FactoresPCA\factores_EXP.txt',delimiter='\t', parse_dates=['_date_'], index_col='_date_').asfreq('MS')
# Unir ambas columnas al DataFrame original
variables_arima_exp_full = variables_arima_exp_full.join([col_pibuem, col_fvirus, col_fcexpor, col_f202004, col_f202005, col_f202006])


train_data_exp = variables_arima_exp_full.loc['1996-01-01':'2024-12-01']
forecast_data_exp = variables_arima_exp_full.loc['1996-01-01':'2025-12-01']


def run_model_exp(df_forecast, numero, y_col, exog_cols=None, order=(1, 0, 0),
                  seasonal_order=(0, 0, 0, 0), trend='n', start_date='2003-04-01',
                  diff=False, diff_exog_cols=None, verbose=True, use_sarimax=False):
    if diff:
        y_full = train_data_exp[y_col].diff()
        y = y_full.loc[start_date:]
    else:
        y = train_data_exp[y_col].loc[start_date:]
    y = y.dropna()

    X = None
    if exog_cols:
        X_raw = train_data_exp[exog_cols].copy()
        if diff_exog_cols:
            for col in diff_exog_cols:
                if col in X_raw.columns:
                    X_raw[col] = X_raw[col].diff()

        X = X_raw.loc[start_date:]
        X = X.loc[y.index]
        X = X[~X.isin([float('inf'), float('-inf')]).any(axis=1) & X.notna().all(axis=1)]
        y = y.loc[X.index]

    ModelClass = sm.tsa.statespace.SARIMAX if use_sarimax else ARIMA
    model = ModelClass(y, exog=X, order=order, seasonal_order=seasonal_order, trend=trend)
    result = model.fit(method='lbfgs', maxiter=1000, disp=False) if use_sarimax else model.fit()

    # if verbose:
    #     print(f"\n--- Modelo fexpp{numero} ---")
    #     print(result.summary())

    steps = 14
    future_dates = pd.date_range(start='2024-10-01', periods=steps,
                                 freq='MS')  # O '2025-01-01' si tienes datos completos

    X_future = None
    if exog_cols:
        extended_dates = pd.date_range(start='2024-10-01', periods=steps + 1, freq='MS')
        X_future_raw = df_forecast.loc[extended_dates, exog_cols].copy()
        if diff_exog_cols:
            for col in diff_exog_cols:
                if col in X_future_raw.columns:
                    X_future_raw[col] = X_future_raw[col].diff()
            X_future = X_future_raw.dropna()
        else:
            X_future = X_future_raw.loc[future_dates]

    forecast = result.forecast(steps=steps, exog=X_future).to_frame(name='predicted_mean')

    if diff:
        last_val = train_data_exp[y_col].loc['2024-10-01']
        forecast['predicted_mean'] = forecast['predicted_mean'].cumsum() + last_val

    col_forecast = f'fexp{numero}f'
    df_forecast = df_forecast.copy()
    df_forecast[col_forecast] = df_forecast[y_col]
    df_forecast.loc[forecast.index, col_forecast] = forecast['predicted_mean']

    serie_completa = df_forecast[col_forecast].loc['2005-03-01':'2025-12-01']
    _, hp_cycle = sm.tsa.filters.hpfilter(serie_completa, lamb=1)

    # Nueva salida: la serie sin HP
    serie_sin_hp = serie_completa.rename(f'FEXP{numero}F')

    return hp_cycle.rename(f'FEXP{numero}F_H'), serie_sin_hp, df_forecast


# Cargar y preparar
variables_arima_exp_full = pd.read_csv(
    r'C:\Users\lucas\High-Frequency_GDP_Forecasting\src\UNION\FactoresPCA\factores_EXP.txt',
    delimiter='\t', parse_dates=['_date_'], index_col='_date_'
).asfreq('MS')

# Cargar ambas columnas desde archivo
col_pibuem = pd.read_csv(r'C:\Users\lucas\High-Frequency_GDP_Forecasting\src\UNION\Exogenas\pibuem.csv', index_col=0, parse_dates=True)
col_fvirus = pd.read_csv(r'C:\Users\lucas\High-Frequency_GDP_Forecasting\src\UNION\Exogenas\fvirus.csv', index_col=0, parse_dates=True)
col_fcexpor = pd.read_csv(r'C:\Users\lucas\High-Frequency_GDP_Forecasting\src\UNION\Exogenas\fcexpor.csv', index_col=0, parse_dates=True)
col_f202004 = pd.read_csv(r'C:\Users\lucas\High-Frequency_GDP_Forecasting\src\UNION\Exogenas\f202004.csv', index_col=0, parse_dates=True)
col_f202005 = pd.read_csv(r'C:\Users\lucas\High-Frequency_GDP_Forecasting\src\UNION\Exogenas\f202005.csv', index_col=0, parse_dates=True)
col_f202006 = pd.read_csv(r'C:\Users\lucas\High-Frequency_GDP_Forecasting\src\UNION\Exogenas\f202006.csv', index_col=0, parse_dates=True)
# Cargar DataFrame base
variables_arima_exp_full = pd.read_csv(r'C:\Users\lucas\High-Frequency_GDP_Forecasting\src\UNION\FactoresPCA\factores_EXP.txt',delimiter='\t', parse_dates=['_date_'], index_col='_date_').asfreq('MS')
# Unir ambas columnas al DataFrame original
variables_arima_exp_full = variables_arima_exp_full.join([col_pibuem, col_fvirus, col_fcexpor, col_f202004, col_f202005, col_f202006])

train_data_exp = variables_arima_exp_full.loc['1996-01-01':'2024-10-01']
forecast_data_exp = variables_arima_exp_full.loc['1996-01-01':'2025-12-01']
forecast_data_exp_mod = forecast_data_exp.copy()

# Ecuación 1
FEXP1F_H, FEXP1F, forecast_data_exp_mod = run_model_exp(
    forecast_data_exp_mod, 1, 'fexp1',
    exog_cols=['pibuem'],
    order=(1, 0, 0), seasonal_order=(1, 0, 0, 12),
    trend='c', start_date='2005-03-01'
)

# Ecuación 2
FEXP2F_H, FEXP2F, forecast_data_exp_mod = run_model_exp(
    forecast_data_exp_mod, 2, 'fexp2',
    exog_cols=['fvirus', 'fcexpor'],
    order=([1, 0, 0, 1], 0, [0, 1, 0, 0, 0, 1]), seasonal_order=(0, 0, 2, 12),
    trend='n', start_date='2005-04-01',
    diff=True
)

# Ecuación 3 (sin exógenas)
FEXP3F_H, FEXP3F, forecast_data_exp_mod = run_model_exp(
    forecast_data_exp_mod, 3, 'fexp3',
    exog_cols=['f202004', 'f202006'],
    order=(2, 0, 1), seasonal_order=(0, 0, 1, 12),
    trend='n', start_date='2005-04-01',
    diff=True
)

# Ecuación 4
FEXP4F_H, FEXP4F, forecast_data_exp_mod = run_model_exp(
    forecast_data_exp_mod, 4, 'fexp4',
    exog_cols=['f202004', 'f202006'],
    order=(2, 0, 2), seasonal_order=(0, 0, 1, 12),
    trend='n', start_date='2005-04-01',
    diff=True, use_sarimax=True
)
# Ecuación 4
FEXP5F_H, FEXP5F, forecast_data_exp_mod = run_model_exp(
    forecast_data_exp_mod, 5, 'fexp5',
    exog_cols=['f202005'],
    order=([0, 1], 0, [0] * 11 + [1]), seasonal_order=(0, 0, 0, 0),
    trend='n', start_date='2005-04-01',
    diff=True
)

output_path_exp = r'C:\Users\lucas\MAF_ESP\Forecast_EXP_HP.csv'
hp_series_exp = pd.concat([
    FEXP1F_H, FEXP2F_H, FEXP3F_H, FEXP4F_H, FEXP5F_H
], axis=1)
hp_series_exp.to_csv(output_path_exp)

# Guardar series sin filtro HP
s_series_exp = pd.concat([
    FEXP1F, FEXP2F, FEXP3F, FEXP4F, FEXP5F
], axis=1)
s_series_exp.to_csv(r'C:\Users\lucas\High-Frequency_GDP_Forecasting\src\UNION\Exogenas_sin_HP\EXP_sin_HP.csv')

# Guardado de series HP
Factores_H_mensuales = pd.concat ([hp_series_agri, hp_series_con, hp_series_cpr, hp_series_cpu, hp_series_exp, hp_series_imp, hp_series_impu, hp_series_ind, hp_series_inv, hp_series_ser], axis=1)
output_path_factores_h_mensuales= r'C:\Users\lucas\High-Frequency_GDP_Forecasting\src\UNION\FactoresPCA\Factores_H_mensuales.csv'
Factores_H_mensuales.to_csv(output_path_factores_h_mensuales)

# Guardado de series sin HP
Factores_S_mensuales_aux = pd.concat ([s_series_agri, s_series_con, s_series_cpr, s_series_cpu, s_series_exp, s_series_imp, s_series_impu, s_series_ind, s_series_inv, s_series_ser], axis=1)

csv_paths = [
    # r'C:\Users\lucas\High-Frequency_GDP_Forecasting\src\UNION\Exogenas\f202311.csv',
    r'C:\Users\lucas\High-Frequency_GDP_Forecasting\src\UNION\Exogenas\f202004.csv',
    r'C:\Users\lucas\High-Frequency_GDP_Forecasting\src\UNION\Exogenas\f202003.csv',
    r'C:\Users\lucas\High-Frequency_GDP_Forecasting\src\UNION\Exogenas\fvirus.csv',
]
dataframes = [pd.read_csv(path, index_col=0, parse_dates=True) for path in csv_paths]

# Concatenar todos (los 10 iniciales + los nuevos)
Factores_S_mensuales = pd.concat([Factores_S_mensuales_aux] + dataframes, axis=1)

# Guardar a CSV
output_path_factores_S_mensuales = r'C:\Users\lucas\High-Frequency_GDP_Forecasting\src\UNION\FactoresPCA\Factores_S_mensuales.csv'
Factores_S_mensuales.to_csv(output_path_factores_S_mensuales)

print("✅ CSV final generado correctamente.")

# Resamplear a trimestral
Factores_H_trimestrales = Factores_H_mensuales.resample('Q').mean()
# Convertir índice a PeriodIndex con frecuencia trimestral
Factores_H_trimestrales.index = pd.PeriodIndex(Factores_H_trimestrales.index, freq='Q')
Factores_H_trimestrales.index.name = '_date_'
# Guardar con índice en formato 2007Q1, etc.
output_path_factores_h_trimestrales = r'C:\Users\lucas\High-Frequency_GDP_Forecasting\src\UNION\FactoresPCA\Factores_H_trimestrales.csv'
Factores_H_trimestrales.to_csv(output_path_factores_h_trimestrales)