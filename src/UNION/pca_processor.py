import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class PCAProcessor:
    def __init__(self, filepath, sheet='LECTURA', start='1995-01-01', end='2025-12-01'):
        self.filepath = filepath
        self.sheet = sheet
        self.start = start
        self.end = end

    @staticmethod
    def dlog(series, lag=0, diff=12):
        return np.log(series.shift(lag)) - np.log(series.shift(lag + diff))

    @staticmethod
    def dl(series, lag=0):
        return series.shift(lag)

    def load_data(self, columns):
        df = pd.read_excel(self.filepath, sheet_name=self.sheet, usecols=columns)
        df.replace('-', np.nan, inplace=True)
        df = df.infer_objects(copy=False).apply(pd.to_numeric, errors='coerce')
        df.index = pd.date_range(start=self.start, end=self.end, freq='MS')
        return df

    def apply_transformations(self, df, fallback_cols=None, lags=(0, 1, 2), diff=12):
        df = df.copy()  # 👈 esto evita SettingWithCopyWarning
        fallback_cols = fallback_cols or []
        base_cols = df.columns.tolist()

        for col in base_cols:
            for lag in lags:
                name = f"DLOG_{col}_{lag}_{diff}"
                if col in fallback_cols:
                    df[name] = self.dl(df[col], lag=lag)
                else:
                    df[name] = self.dlog(df[col], lag=lag, diff=diff)

        return df.iloc[:, len(base_cols):]  # sólo columnas transformadas

    def extract_valid_block(self, df):
        valid_rows = ~df.isnull().any(axis=1)
        start_pos = valid_rows.values.argmax()
        end_pos = next((i for i in range(start_pos, len(valid_rows)) if not valid_rows.iloc[i]), len(valid_rows))
        return df.iloc[start_pos:end_pos]

    def pca_pipeline(self, df, n_components, fecha_corte, prefix):
        df_valid = self.extract_valid_block(df)
        df_filtered = df_valid.loc[:fecha_corte]
        data_scaled = StandardScaler().fit_transform(df_filtered)
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(data_scaled)
        columns = [f'{prefix}{i+1}' for i in range(n_components)]
        return pd.DataFrame(X_pca, index=df_filtered.index, columns=columns)

    def run(self, columns, fallback_cols, n_components, fecha_corte, prefix):
        df = self.load_data(columns)
        df_transformed = self.apply_transformations(df, fallback_cols=fallback_cols)
        return self.pca_pipeline(df_transformed, n_components, fecha_corte, prefix)
