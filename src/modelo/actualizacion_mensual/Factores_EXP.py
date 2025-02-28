import pandas as pd
import statsmodels.api as sm
import pandas as pd
import os
from src.__special__ import indices_path

# Cargar los datos (debes reemplazar 'data.csv' con tu archivo real)

file_path = os.path.join(indices_path, 'Factores_EXP.txt')
Factores_EXP = pd.read_csv(file_path,delimiter="\t")

print(Factores_EXP)