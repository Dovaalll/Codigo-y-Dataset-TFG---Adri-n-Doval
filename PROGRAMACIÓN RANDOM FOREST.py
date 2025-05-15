# Carga y Preprocesamiento de Datos
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Cargar el dataset limpio
casas = pd.read_csv("DATASETS/casas_baratas_normalizado.csv", delimiter=";")

# Definir Variables Predictoras (X) y Variable Objetivo (y)
X = casas.drop(columns=["PRICE"])
y = casas["PRICE"]

# División de Datos en Entrenamiento y Prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Construcción y Entrenamiento del Modelo Random Forest
rf_model = RandomForestRegressor(n_estimators=100, max_depth=None, min_samples_split=2, random_state=42)
rf_model.fit(X_train, y_train)

# Predicción y Evaluación del Modelo
y_pred = rf_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("Evaluación del Modelo Random Forest:")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.4f}")

# Visualización de Resultados
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.5, color="blue")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red')
plt.xlabel("Valores Reales")
plt.ylabel("Valores Predichos")
plt.title("Comparación entre Predicción y Valor Real (Random Forest)")
plt.show()

# Análisis de Importancia de Variables
importances = rf_model.feature_importances_
variables_importantes = pd.DataFrame({'Variable': X.columns, 'Importancia': importances})
variables_importantes = variables_importantes.sort_values(by='Importancia', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x="Importancia", y="Variable", data=variables_importantes, palette="viridis")
plt.xlabel("Importancia")
plt.ylabel("Variables")
plt.title("Importancia de Variables en Random Forest")
plt.show()

# Guardado del Modelo
joblib.dump(rf_model, "random_forest_model.pkl")
print("Modelo guardado como 'random_forest_model.pkl'")
