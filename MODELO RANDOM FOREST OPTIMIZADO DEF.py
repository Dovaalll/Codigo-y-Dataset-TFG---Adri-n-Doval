import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Cargar el dataset limpio
ruta_dataset = "DATASETS/DATASET_CASAS_PROCESADO.csv"
casas = pd.read_csv(ruta_dataset, delimiter=";")

# **Transformación Logarítmica en el Precio**
casas["PRICE"] = np.log1p(casas["PRICE"])  # log(1 + PRICE) para evitar valores negativos

# Separar variables predictoras y objetivo
X = casas.drop(columns=["PRICE"])
y = casas["PRICE"]

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# **1. Selección de Variables Basada en Importancia**
modelo_prueba = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
modelo_prueba.fit(X_train, y_train)
importancias = modelo_prueba.feature_importances_

# Crear DataFrame con las importancias
importancia_vars = pd.DataFrame({"Variable": X_train.columns, "Importancia": importancias})
importancia_vars = importancia_vars.sort_values(by="Importancia", ascending=False)

# Seleccionar las variables más importantes (las que sumen al menos 95% de la importancia acumulada)
importancia_vars["Acumulado"] = importancia_vars["Importancia"].cumsum()
variables_seleccionadas = importancia_vars[importancia_vars["Acumulado"] <= 0.95]["Variable"].tolist()

# Filtrar dataset con las variables más importantes
X_train = X_train[variables_seleccionadas]
X_test = X_test[variables_seleccionadas]

# **2. Optimización de Hiperparámetros con GridSearchCV**
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [10, 20, 30, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"],
    "bootstrap": [True, False]
}

# Definir modelo con GridSearch
grid_search = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42, oob_score=True, n_jobs=-1),
    param_grid=param_grid,
    cv=3,
    verbose=2
)

# Ajustar el modelo con la búsqueda en cuadrícula
grid_search.fit(X_train, y_train)

# Obtener los mejores hiperparámetros
mejores_params = grid_search.best_params_
print(f"Mejores Hiperparámetros: {mejores_params}")

# **3. Entrenar el Modelo con los Mejores Parámetros**
modelo_rf = RandomForestRegressor(**mejores_params, random_state=42, oob_score=True, n_jobs=-1)
modelo_rf.fit(X_train, y_train)

# **4. Evaluación del Modelo**
y_pred = modelo_rf.predict(X_test)

# **Revertir la transformación logarítmica para obtener valores en euros**
y_test_real = np.expm1(y_test)
y_pred_real = np.expm1(y_pred)

# Cálculo de métricas corregido
r2 = r2_score(y_test_real, y_pred_real)
rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
mae = mean_absolute_error(y_test_real, y_pred_real)
rae = np.sum(np.abs(y_test_real - y_pred_real)) / np.sum(np.abs(y_test_real - np.mean(y_test_real)))

# Mostrar resultados
print(f"\n📊 Evaluación del Modelo Random Forest Optimizado:")
print(f"  🔹 R² Score: {r2:.4f}")
print(f"  🔹 RMSE (Raíz del Error Cuadrático Medio): {rmse:.2f}€")
print(f"  🔹 MAE (Error Absoluto Medio): {mae:.2f}€")
print(f"  🔹 RAE (Relative Absolute Error): {rae:.4f}")
print(f"  🔹 OOB Score (Validación Interna OOB): {modelo_rf.oob_score_:.4f}")

# **5. Visualización de Resultados**
# Gráfico de Importancia de Variables
plt.figure(figsize=(10, 6))
sns.barplot(x="Importancia", y="Variable", data=importancia_vars, palette="mako")
plt.title("Importancia de Variables en Random Forest Optimizado")
plt.show()

# Gráfico de comparación entre predicciones y valores reales
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test_real, y=y_pred_real, alpha=0.5, color="blue")
plt.plot([min(y_test_real), max(y_test_real)], [min(y_test_real), max(y_test_real)], "--r")
plt.xlabel("Valores Reales (€)")
plt.ylabel("Valores Predichos (€)")
plt.title("Comparación entre Predicción y Valor Real (Random Forest)")
plt.show()
