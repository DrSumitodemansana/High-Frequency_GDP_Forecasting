from pca_processor import PCAProcessor
import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

# Ruta al Excel
filepath = r'C:\Users\lucas\High-Frequency_GDP_Forecasting\src\data\AFESP.xlsx'
processor = PCAProcessor(filepath)

# Diccionario de configuraciones
bloques = {
    "CP": dict(cols=[f'CP{i}' for i in range(1, 22)], fallback=['CP16', 'CP18'], n=7, corte='2024-12-01'),
    "EXP": dict(cols=[f'EXP{i}' for i in range(1, 13)], fallback=[], n=5, corte='2024-11-01'),
    "CPU": dict(cols=[f'CPU{i}' for i in range(1, 5)], fallback=[], n=3, corte='2024-12-01'),
    "INV": dict(cols=[f'INV{i}' for i in range(1, 10)], fallback=['INV9'], n=5, corte='2024-12-01'),
    "IMP": dict(cols=[f'IMP{i}' for i in range(1, 5)] + ['EXP5', 'EXP6'], fallback=[], n=4, corte='2024-10-01'),
    "AGR": dict(cols=[f'AG{i}' for i in range(1, 9)], fallback=[], n=5, corte='2024-12-01'),
    "CON": dict(cols=[f'CON{i}' for i in range(1, 17)], fallback=['CON9', 'CON10', 'CON11', 'CON14'], n=5, corte='2024-12-01'),
    "SER": dict(cols=[f'SER{i}' for i in range(1, 24)], fallback=['SER17', 'SER19', 'SER20', 'SER21'], n=6, corte='2024-12-01'),
    "IMPU": dict(cols=[f'IMPU{i}' for i in range(1, 4)], fallback=[], n=4, corte='2024-10-01'),
}

# Proceso regular
factores = {}
for key, config in bloques.items():
    factores[key] = processor.run(
        columns=config['cols'],
        fallback_cols=config['fallback'],
        n_components=config['n'],
        fecha_corte=config['corte'],
        prefix=f"f{key.lower()}"
    )

# IND requiere tratamiento especial
ind_cols = [f'IND{i}' for i in range(1, 14)]
df_ind = processor.load_data(ind_cols)
df_ind['DLOG_IND11_0_12'] = (np.log(df_ind['IND11']) - np.log(df_ind['IND11'].shift(12))) - \
                            (np.log(df_ind['IND11'].shift(1)) - np.log(df_ind['IND11'].shift(13)))
for lag in [1, 2]:
    df_ind[f'DLOG_IND11_{lag}_12'] = df_ind['DLOG_IND11_0_12'].shift(lag)
rest = [col for col in ind_cols if col != 'IND11']
df_ind_transformed = processor.apply_transformations(df_ind[rest], fallback_cols=['IND7', 'IND8', 'IND9'])
df_ind_full = pd.concat([df_ind_transformed, df_ind[[f'DLOG_IND11_{i}_12' for i in [0, 1, 2]]]], axis=1)
factores["IND"] = processor.pca_pipeline(df_ind_full, n_components=6, fecha_corte='2024-12-01', prefix='find')

# Crear índice base completo
full_index = pd.date_range(start='1995-01-01', end='2026-12-01', freq='MS')

ruta_salida = r'C:\Users\lucas\High-Frequency_GDP_Forecasting\src\UNION\FactoresPCA'

# Guardar cada grupo de factores en formato .txt con índice completo y columna '_date_'
for key, df in factores.items():
    df_out = df.reindex(full_index)
    df_out.index.name = '_date_'
    output_path = os.path.join(ruta_salida, f'factores_{key}.txt')
    df_out.to_csv(output_path, sep='\t', float_format='%.6f', na_rep='NaN')

