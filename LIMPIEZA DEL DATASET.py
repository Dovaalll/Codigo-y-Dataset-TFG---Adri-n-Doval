import pandas as pd
import numpy as np
from scipy.stats import zscore

# **1. Cargar el dataset**
ruta_dataset = "DATASETS/DATASET_CASAS.csv"
casas = pd.read_csv(ruta_dataset, delimiter=";")

# **2. Eliminación de Variables Irrelevantes**
variables_a_eliminar = [
    "ID", "HAS_NORTH_ORIENTATION", "HAS_SOUTH_ORIENTATION", "HAS_EAST_ORIENTATION",
    "HAS_WEST_ORIENTATION", "BUILTTYPEID_1", "BUILTTYPEID_2", "BUILTTYPEID_3",
    "AMENITY", "CADDWELLINGCOUNT", "DISTANCE_TO_METRO"
]
casas.drop(columns=variables_a_eliminar, inplace=True)

# **3. Eliminación de Outliers con Z-score**
def eliminar_outliers_zscore(df, columnas, umbral=3):
    df_sin_outliers = df[(np.abs(zscore(df[columnas])) < umbral).all(axis=1)]
    return df_sin_outliers

columnas_outliers = ["PRICE", "CONSTRUCTED_AREA", "DISTANCE_TO_CITY_CENTER"]
casas = eliminar_outliers_zscore(casas, columnas_outliers)

# **4. Verificación y eliminación de valores nulos**
casas.dropna(inplace=True)

# **5. Guardar el dataset limpio**
ruta_guardado = "DATASETS/DATASET_CASAS_LIMPIO.csv"
casas.to_csv(ruta_guardado, index=False, sep=";")

print("Limpieza de datos completada. Dataset guardado en:", ruta_guardado)
