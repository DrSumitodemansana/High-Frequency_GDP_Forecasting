import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
import win32com.client
import pandas as pd
import time

warnings.filterwarnings('ignore', category=FutureWarning)
# Cargar datos de Factores_h_trimestrales y EXCEL

# FACTORES_H_TRIMESTRALES
factores_h_trimestral = pd.read_csv(r'C:\Users\lucas\High-Frequency_GDP_Forecasting\src\UNION\FactoresPCA\Factores_H_trimestrales.csv', delimiter=',')
factores_h_trimestral['_date_'] = pd.PeriodIndex(factores_h_trimestral['_date_'], freq='Q')
factores_h_trimestral.set_index('_date_', inplace=True)
factores_h_trimestral = factores_h_trimestral.loc['2007Q1':'2024Q3']

factores_h_trimestral['uno'] = 1

# EXCEL
def extraer_hoja_excel_con_valores(ruta_excel, hoja_original="PIB", hoja_valores="PIB_Valores", header=1):
    # Lanzar Excel
    excel = win32com.client.Dispatch("Excel.Application")
    excel.Visible = False
    excel.DisplayAlerts = False

    # Abrir libro con actualización de vínculos
    wb = excel.Workbooks.Open(Filename=ruta_excel, UpdateLinks=3)
    time.sleep(2)

    # Eliminar hoja temporal si existe
    try:
        wb.Sheets(hoja_valores).Delete()
    except:
        pass

    # Copiar hoja original y pegar solo valores
    ws_original = wb.Sheets(hoja_original)
    ws_original.Copy(After=wb.Sheets(wb.Sheets.Count))
    ws_valores = wb.Sheets(wb.Sheets.Count)
    ws_valores.Name = hoja_valores
    ws_valores.Cells.Copy()
    ws_valores.Cells.PasteSpecial(Paste=-4163)  # Pegado solo valores

    # Guardar y cerrar
    wb.Save()
    wb.Close(SaveChanges=True)
    excel.Quit()

    # Leer hoja pegada desde pandas
    return pd.read_excel(ruta_excel, sheet_name=hoja_valores, header=header)

def preparar_componentes_pib(df_raw):
    # Renombrar y transformar índice
    df_raw.rename(columns={df_raw.columns[0]: "_date_"}, inplace=True)
    df_raw.set_index("_date_", inplace=True)
    df_raw.index = pd.PeriodIndex(df_raw.index.astype(str), freq='Q')

    # Seleccionar solo las primeras 11 columnas
    df = df_raw.iloc[:, :11].copy()
    df["uno"] = 1

    # Crear columnas PCY_ con variación interanual
    columnas_numericas = [col for col in df.columns if col != "uno"]
    columnas_pcy = {col: f"PCY_{col}" for col in columnas_numericas}
    df = df.assign(**{
        new_col: df[old_col].pct_change(periods=4) * 100
        for old_col, new_col in columnas_pcy.items()
    })

    # Filtrar el rango deseado
    df = df.loc["2007Q1":"2024Q3"]

    return df

# === RUTAS ===
ruta_excel = r"C:\Users\lucas\High-Frequency_GDP_Forecasting\src\modelo\AFESP.xlsx"
output_csv = r"C:\Users\lucas\High-Frequency_GDP_Forecasting\src\UNION\FactoresPCA\ComponentesPIB.csv"

# === FLUJO PRINCIPAL ===
componentespib_raw = extraer_hoja_excel_con_valores(ruta_excel)
componentespib = preparar_componentes_pib(componentespib_raw)
componentespib.to_csv(output_csv, sep=';', decimal='.')


# Diccionario de variables exógenas específicas para cada regresión
variables_exogenas = {
    "PCY_CPR": ["uno", "FCP1F_H", "FCP2F_H", "FCP3F_H", "FCP4F_H", "FCP5F_H", "FCP6F_H", "FCP7F_H"],
    "PCY_CPU": ["uno", "FCPU1F_H", "FCPU2F_H", "FCPU3F_H"],
    "PCY_INV": ["uno", "FINV1F_H", "FINV2F_H", "FINV3F_H", "FINV4F_H", "FINV5F_H"],
    "PCY_EXPOR": ["uno", "FEXP1F_H", "FEXP2F_H", "FEXP3F_H", "FEXP4F_H"],
    "PCY_IMP": ["uno", "FIMP1F_H", "FIMP2F_H", "FIMP3F_H", "FIMP4F_H"],
    "PCY_AGRICUL": ["uno", "FAGR1F_H", "FAGR2F_H", "FAGR3F_H", "FAGR4F_H", "FAGR5F_H"],
    "PCY_INDUSTRIA": ["uno", "FIND1F_H", "FIND2F_H", "FIND3F_H", "FIND4F_H", "FIND5F_H", "FIND6F_H"],
    "PCY_CONSTRU": ["uno", "FCON1F_H", "FCON2F_H", "FCON3F_H", "FCON4F_H", "FCON5F_H"],
    "PCY_SERVI": ["uno", "FSER1F_H", "FSER2F_H", "FSER3F_H", "FSER4F_H", "FSER5F_H", "FSER6F_H"],
    "PCY_IMPUESTOS": ["uno", "FIMPU1F_H", "FIMPU2F_H", "FIMPU3F_H", "FIMPU4F_H"]
}

# Diccionario para almacenar los modelos
resultados_modelos = {}

# Ajustar modelos de regresión OLS
for columna, predictores in variables_exogenas.items():
    if columna in componentespib.columns:
        # Verificar que las variables exógenas existen en factores_h_trimestral
        factores_disponibles = [var for var in predictores if var in factores_h_trimestral.columns]

        if not factores_disponibles:
            print(f"⚠️ No hay variables exógenas disponibles para {columna}. Modelo omitido.")
            continue

        # Variables dependiente (PCY_*)
        y = componentespib[columna]

        # Variables exógenas (seleccionadas según la ecuación)
        X = factores_h_trimestral[factores_disponibles]

        # Asegurar que ambos DataFrames tengan los mismos índices
        y, X = y.align(X, join="inner", axis=0)

        # Ajustar el modelo OLS
        modelo = sm.OLS(y, X).fit()

        # Guardar el modelo en el diccionario
        resultados_modelos[columna] = modelo

# Diccionario para almacenar sigma y ro de cada modelo
resultados_estadisticos = {}

for columna, modelo in resultados_modelos.items():
    sigma = modelo.mse_resid
    ro = 1 - (sm.stats.durbin_watson(modelo.resid) / 2)
    resultados_estadisticos[columna] = {"sigma": sigma, "ro": ro}

# Convertir resultados_estadisticos en un DataFrame
df_estadisticos = pd.DataFrame(resultados_estadisticos).T

# Crear un diccionario para almacenar los residuos de cada modelo
residuos_dict = {}

for columna, modelo in resultados_modelos.items():
    residuos = modelo.resid
    residuos_dict[columna] = residuos

# Convertir los residuos a un DataFrame
df_residuos = pd.DataFrame(residuos_dict)

# Cargar datos mensuales
x_prueba = pd.read_csv(r'C:\Users\lucas\High-Frequency_GDP_Forecasting\src\UNION\FactoresPCA\Factores_H_mensuales.csv', delimiter=',')
_date_ = pd.date_range(start="1996-03-01", end="2025-12-01", freq="MS")
x_prueba.index = _date_
x_prueba.index.name = '_date_'
x_prueba = x_prueba.drop(columns='_date_')
x_prueba['uno'] = 1
x_prueba = x_prueba.loc['2007-01-01':'2025-12-01']

# Crear diccionario para almacenar resultados de Ymens
Ymens_dict = {}

# Iterar sobre todas las variables en df_estadisticos
for variable in df_estadisticos.index:
    sigma_var = df_estadisticos.loc[variable, 'sigma']
    ro_var = df_estadisticos.loc[variable, 'ro']

    # Extraer Y e X para la variable actual
    y_impu = componentespib[variable]
    x_impu = factores_h_trimestral[variables_exogenas[variable]]

    filas = len(y_impu)
    filasa = filas + 5  # Ajustar con t

    V3 = np.ones(3) * (1/3)
    B = np.kron(np.eye(filas), V3.reshape(1, -1))
    BA = np.kron(np.eye(filasa), V3.reshape(1, -1))

    if ro_var > 0:
        rot_var = np.exp(-0.00302073936725
                 + 0.459080195566 * np.log(ro_var)
                 - 0.0517950701174 * np.log(ro_var)**2
                 + 0.0114881891532 * np.log(ro_var)**3
                 + 0.0019942146442 * np.log(ro_var)**4
                 + 4.58020371897e-05 * np.log(ro_var)**5)

    M1 = np.identity(3 * filas)
    M1a = np.identity(3 * filasa)

    for i in range(1, 3 * filasa):
        RHOa = np.full((3 * filasa - i,), rot_var ** i)
        M1a += np.diag(RHOa, -i) + np.diag(RHOa, i)

    for i in range(1, 3 * filas):
        rho = rot_var ** i * np.ones(3 * filas - i)
        np.fill_diagonal(M1[i:, :-i], rho)
        np.fill_diagonal(M1[:-i, i:], rho)

    if rot_var > 0:
        factor = sigma_var / (1 - rot_var**2)
        vMENS = M1 * factor
        vMENSa = M1a * factor

    V = B @ vMENS @ B.T
    VA = BA @ vMENSa @ BA.T

    x_impu = x_impu.to_numpy()
    y_impu = y_impu.to_numpy().reshape(-1, 1)
    BETAG = np.linalg.inv(x_impu.T @ x_impu) @ (x_impu.T @ y_impu)

    ERRmens = y_impu - (x_impu @ BETAG)
    AJUSTE = ERRmens[-1]

    ERRMENSA = np.zeros((filasa, 1))
    ERRMENSA[:filas, :] = ERRmens
    ERRMENSA[filas:, :] = AJUSTE

    Ymens_dict[variable] = x_prueba[variables_exogenas[variable]] @ BETAG + vMENSa @ (BA.T @ np.linalg.inv(VA) @ ERRMENSA)

# Convertir el diccionario Ymens_dict a un DataFrame
Ymens_final = pd.DataFrame({var: Ymens_dict[var].to_numpy().flatten() for var in Ymens_dict},
                           index=x_prueba.index)

Ymens_final.columns = Ymens_final.columns.str.replace("PCY_", "", regex=True)

# 4. Generación del PIB mensual
Ymens_final['PIB_Demanda_mensual'] = (0.596 * Ymens_final['CPR'] +
                              0.166 * Ymens_final['CPU'] +
                              0.267 * Ymens_final['INV'] +
                              0.286 * Ymens_final['EXPOR'] -
                              0.315 * Ymens_final['IMP'])

Ymens_final['PIB_Oferta_mensual'] = (Ymens_final['AGRICUL'] * 0.0249973 +
                             Ymens_final['INDUSTRIA'] * 0.1472338 +
                             Ymens_final['CONSTRU'] * 0.0539303 +
                             Ymens_final['SERVI'] * 0.6866292 +
                             Ymens_final['IMPUESTOS'] * 0.0872094)

# Mostrar el resultado
Ymens_final.to_csv('Ymens_final.csv', sep=';', decimal='.', index=True)