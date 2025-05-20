import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Cargar el dataset completo
ruta_dataset = "DATASETS/DATASET_CASAS_LIMPIO.csv"
casas = pd.read_csv(ruta_dataset, delimiter=";")

# Función para eliminar outliers con IQR (Rango Intercuartílico)
def eliminar_outliers_iqr(df, columnas):
    df_limpio = df.copy()
    for col in columnas:
        Q1 = df_limpio[col].quantile(0.25)
        Q3 = df_limpio[col].quantile(0.75)
        IQR = Q3 - Q1
        filtro = (df_limpio[col] >= (Q1 - 1.5 * IQR)) & (df_limpio[col] <= (Q3 + 1.5 * IQR))
        df_limpio = df_limpio[filtro]
    return df_limpio

# Aplicar eliminación de atípicos en las variables clave
columnas_outliers = ["PRICE", "CONSTRUCTED_AREA", "ROOM_NUMBER", "BATH_NUMBER"]
casas_sin_outliers = eliminar_outliers_iqr(casas, columnas_outliers)

# Normalización de las variables numéricas
scaler = MinMaxScaler()
columnas_numericas = casas_sin_outliers.select_dtypes(include=['float64', 'int64']).columns.tolist()
columnas_numericas.remove("PRICE")  # No normalizamos la variable objetivo

casas_sin_outliers[columnas_numericas] = scaler.fit_transform(casas_sin_outliers[columnas_numericas])

# Guardar el dataset procesado
ruta_guardado = "DATASETS/DATASET_CASAS_PROCESADO.csv"
casas_sin_outliers.to_csv(ruta_guardado, index=False, sep=";")

# Mostrar resultados
print(f"Dataset original: {len(casas)} registros")
print(f"Dataset sin outliers: {len(casas_sin_outliers)} registros")
print(f"Dataset guardado como: {ruta_guardado}")
