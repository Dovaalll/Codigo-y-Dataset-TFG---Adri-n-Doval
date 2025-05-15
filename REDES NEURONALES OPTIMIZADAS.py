import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import gc
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Liberar memoria antes de ejecutar el código
gc.collect()
torch.cuda.empty_cache()

# Cargar el dataset normalizado
ruta_dataset = "DATASETS/DATASET_CASAS_PROCESADO.csv"
df = pd.read_csv(ruta_dataset, delimiter=";")

# Separar variables predictoras y variable objetivo
X = df.drop(columns=["PRICE"]).values
y = df["PRICE"].values.reshape(-1, 1)

# Normalizar los datos
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y)

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convertir a tensores de PyTorch
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Definir el modelo de red neuronal
class RedNeuronal(nn.Module):
    def __init__(self, input_dim):
        super(RedNeuronal, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)  # Salida sin activación para regresión
        return x

# Inicializar el modelo
input_dim = X_train.shape[1]
modelo = RedNeuronal(input_dim)

# Definir función de pérdida y optimizador
criterio = nn.HuberLoss()
optimizador = optim.Adam(modelo.parameters(), lr=0.001)

# Definir early stopping manual
paciencia = 50
mejor_pérdida = float("inf")
paciencia_actual = 0

# Entrenamiento del modelo
num_epocas = 1000
batch_size = 16
perdidas = []

print("🔄 Iniciando entrenamiento...")
for epoch in range(num_epocas):
    modelo.train()
    indices = torch.randperm(X_train_tensor.shape[0])
    batch_losses = []

    for i in range(0, X_train_tensor.shape[0], batch_size):
        indices_batch = indices[i:i + batch_size]
        batch_X = X_train_tensor[indices_batch]
        batch_y = y_train_tensor[indices_batch]

        optimizador.zero_grad()
        predicciones = modelo(batch_X)
        loss = criterio(predicciones, batch_y)
        loss.backward()
        optimizador.step()

        batch_losses.append(loss.item())

    perdida_epoch = np.mean(batch_losses)
    perdidas.append(perdida_epoch)

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {perdida_epoch:.6f}")

    # Early stopping
    if perdida_epoch < mejor_pérdida:
        mejor_pérdida = perdida_epoch
        paciencia_actual = 0
    else:
        paciencia_actual += 1
        if paciencia_actual >= paciencia:
            print(f"⏹️ Early Stopping en la época {epoch}")
            break

print("✅ Entrenamiento completado.")

# Evaluación del modelo
modelo.eval()
with torch.no_grad():
    y_pred_tensor = modelo(X_test_tensor)
    y_pred = y_pred_tensor.numpy()
    y_pred = scaler_y.inverse_transform(y_pred)  # Desnormalizar
    y_test = scaler_y.inverse_transform(y_test)  # Desnormalizar

# Calcular métricas de error
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

# Mostrar resultados
print("\n📊 Evaluación del Modelo Optimizado:")
print(f"🔹 R² Score: {r2:.4f}")
print(f"🔹 MSE: {mse:.2f}€")
print(f"🔹 RMSE: {rmse:.2f}€")
print(f"🔹 MAE: {mae:.2f}€")
print(f"🔹 MAPE: {mape:.2f}%")

# Graficar evolución de la pérdida
plt.figure(figsize=(8, 5))
plt.plot(perdidas, label="Pérdida (Huber Loss)")
plt.xlabel("Épocas")
plt.ylabel("Pérdida")
plt.title("Evolución de la Pérdida Durante el Entrenamiento")
plt.legend()
plt.show()

# Graficar predicción vs. valores reales
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color="blue")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "--", color="red")
plt.xlabel("Valores Reales (€)")
plt.ylabel("Valores Predichos (€)")
plt.title("Comparación entre Predicción y Valor Real (Red Neuronal Optimizada)")
plt.show()

# Liberar memoria después de ejecutar
gc.collect()
torch.cuda.empty_cache()
