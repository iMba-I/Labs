import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

# Класс NeuralNetwork без изменений (тот же, что и раньше)
class NeuralNetwork:
    def __init__(self, input_size, architecture, output_size=1):
        layer_sizes = [input_size] + architecture + [output_size]
        self.L = len(layer_sizes) - 1

        self.weights = []
        self.biases = []
        for i in range(self.L):
            W = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.01
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(W)
            self.biases.append(b)

        
        self.train_mse = []
        self.train_mae = []
        self.train_r2 = []

    def forward(self, X):
        activations = [X]
        zs = []

        A = X
        for i in range(self.L):
            Z = A @ self.weights[i] + self.biases[i]
            A = Z
            zs.append(Z)
            activations.append(A)

        return A, zs, activations

    def backward(self, y, zs, activations):
        m = y.shape[0]
        y = y.reshape(-1, 1)

        grads_W = [None] * self.L
        grads_b = [None] * self.L

        dA = (2 / m) * (activations[-1] - y)

        for i in reversed(range(self.L)):
            A_prev = activations[i]
            dZ = dA

            grads_W[i] = A_prev.T @ dZ
            grads_b[i] = np.sum(dZ, axis=0, keepdims=True)
            dA = dZ @ self.weights[i].T

        return grads_W, grads_b

    def update_params(self, grads_W, grads_b, lr):
        for i in range(self.L):
            self.weights[i] -= lr * grads_W[i]
            self.biases[i]  -= lr * grads_b[i]

    def predict(self, X):
        y_pred, _, _ = self.forward(X)
        return y_pred.ravel()

    def fit(self, X, y, lr=0.00001, epochs=200, batch_size=32):
        m = X.shape[0]

        for epoch in range(1, epochs + 1):
            perm = np.random.permutation(m)
            X_shuffled = X[perm]
            y_shuffled = y[perm]

            for start in range(0, m, batch_size):
                end = min(start + batch_size, m)
                xb = X_shuffled[start:end]
                yb = y_shuffled[start:end]

                y_pred, zs, activations = self.forward(xb)
                grads_W, grads_b = self.backward(yb, zs, activations)
                self.update_params(grads_W, grads_b, lr)

            # Вычисляем метрики на всей тренировочной выборке
            y_train_pred = self.predict(X)
            mse = mean_squared_error(y, y_train_pred)
            mae = mean_absolute_error(y, y_train_pred)
            r2 = r2_score(y, y_train_pred)

            self.train_mse.append(mse)
            self.train_mae.append(mae)
            self.train_r2.append(r2)

            if epoch % 10 == 0 or epoch == 1:
                print(f"Epoch {epoch}: MSE={mse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")


# Загрузка и подготовка данных
data = fetch_california_housing()
X, y = data.data, data.target
feature_names = data.feature_names

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Создаем модель
nn = NeuralNetwork(
    input_size=X.shape[1],
    architecture=[20, 10],
    output_size=1
)

# Обучение
nn.fit(
    X_train, y_train,
    lr=0.0005,
    epochs=200,
    batch_size=32
)

# Предсказания на обучении
y_train_pred = nn.predict(X_train)

# Визуализация: 8 графиков по признакам
plt.figure(figsize=(20, 15))

for i, feature in enumerate(feature_names):
    plt.subplot(3, 3, i + 1)
    plt.scatter(X_train[:, i], y_train, label='Реальные', alpha=0.5, s=15)
    plt.scatter(X_train[:, i], y_train_pred, label='Предсказанные', alpha=0.5, s=15)
    plt.xlabel(feature)
    plt.ylabel('Целевое значение')
    plt.title(f'Зависимость целевого от {feature}')
    plt.legend()

plt.tight_layout()
plt.show()

print("Первые веса первого слоя после обучения:\n", nn.weights[0])
plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)
plt.plot(range(1, len(nn.train_mse) + 1), nn.train_mse)
plt.title('Train MSE')
plt.xlabel('Epoch')
plt.grid(True)

# --- 2. MAE ---
plt.subplot(1, 3, 2)
plt.plot(range(1, len(nn.train_mae) + 1), nn.train_mae)
plt.title('Train MAE')
plt.xlabel('Epoch')
plt.grid(True)

# --- 3. R² ---
plt.subplot(1, 3, 3)
plt.plot(range(1, len(nn.train_r2) + 1), nn.train_r2)
plt.title('Train R²')
plt.xlabel('Epoch')
plt.grid(True)

plt.tight_layout()
plt.show()