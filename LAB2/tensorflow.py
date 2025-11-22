import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow import keras

# ======================================
# Загрузка и подготовка данных
# ======================================
data = fetch_california_housing()
X, y = data.data, data.target

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ======================================
# Модель Keras (линейные слои без активаций)
# ======================================
model = keras.Sequential([
    keras.layers.Dense(20, activation=None, input_shape=(X.shape[1],)),
    keras.layers.Dense(10, activation=None),
    keras.layers.Dense(1, activation=None)
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

# ======================================
# Оценка
# ======================================
y_pred = model.predict(X_test).ravel()

print("TF MSE:", mean_squared_error(y_test, y_pred))
print("TF MAE:", mean_absolute_error(y_test, y_pred))
print("TF R² :", r2_score(y_test, y_pred))

# ======================================
# Графики обучения
# ======================================
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train MSE')
plt.plot(history.history['val_loss'], label='Test MSE')
plt.legend()
plt.title("TensorFlow MSE")

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Test MAE')
plt.legend()
plt.title("TensorFlow MAE")

plt.tight_layout()
plt.show()
