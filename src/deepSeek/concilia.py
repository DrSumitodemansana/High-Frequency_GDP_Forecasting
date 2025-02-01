import pandas as pd
import numpy as np
import os
from src.__special__ import data_path
from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.tsa.holtwinters import ExponentialSmoothing



# 2. Definir funciones para manipulación de datos y modelado (hecho dentro del codigo)


def concilia(data):
    # 3. Procesamiento de datos y creación de variables
    anos_c = 17  # Número de años completos
    an = 19      # Número de años disponibles
    tri = an * 4
    mes = an * 12
    ser = 10
    p_ofe = 0.5
    p_dem = 1 - p_ofe

    v4 = np.ones(4)
    v12 = np.ones(12)
    htw = np.zeros((tri, tri * ser))
    hmw = np.zeros(((tri * (ser - 1)) + mes, mes * ser))

    # 4. Aplicar filtro HP a las series temporales
    for col in ['M_CPR', 'M_CPU', 'M_INV', 'M_EXP', 'M_IMP', 'M_AGRI', 'M_INDU', 'M_CST', 'M_IMPU', 'M_SER']:
        cycle, data[f'{col}_H'] = hpfilter(data[col], lamb=2)

    # 5. Crear grupos de datos
    M_ini = data[['M_CPR_H', 'M_CPU_H', 'M_INV_H', 'M_EXP_H', 'M_IMP_H', 'I_M_AGRI_H', 'I_M_INDU_H', 'I_M_CST_H', 'I_M_IMPU_H', 'I_M_SER_H']]
    M_inir = data[['M_CPR', 'M_CPU', 'M_INV', 'M_EXP', 'M_IMP', 'I_M_AGRI', 'I_M_INDU', 'I_M_CST', 'I_M_IMPU', 'I_M_SER']]

    # 6. Convertir grupos a matrices
    M_Y_ini = M_ini.values
    M_Y_inir = M_inir.values

    # 7. Producto de Kronecker
    Bagre = np.kron(np.identity(tri), np.ones(3) / 3)

    # 8. Operaciones matriciales
    T_Y_ini = Bagre @ M_Y_inir
    VT_Y_ini = T_Y_ini.flatten()
    dem_T_ini = VT_Y_ini[:tri * ser // 2]
    ofe_T_ini = -VT_Y_ini[tri * ser // 2:]

    # 9. Exportar resultados
    finales = pd.DataFrame({
        'MF_CPR': M_Y_ini[:, 0],
        'MF_CPU': M_Y_ini[:, 1],
        'MF_INV': M_Y_ini[:, 2],
        'MF_EXP': M_Y_ini[:, 3],
        'MF_IMP': M_Y_ini[:, 4],
        'MF_AGRI': -M_Y_ini[:, 5],
        'MF_INDU': -M_Y_ini[:, 6],
        'MF_CST': -M_Y_ini[:, 7],
        'MF_IMPU': -M_Y_ini[:, 8],
        'MF_SER': -M_Y_ini[:, 9]
    })

    finales.to_excel("prediccion.xlsx", index=False)


if __name__ == "__main__":
    file_name = os.path.join(data_path, "AFESP.xlsx")
    # 1. Importar datos
    data = pd.read_excel("AFESP.xlsx", sheet_name="Nominal_a")
    concilia(data)

