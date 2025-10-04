import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

# ========================
# 1. Генерация датасета
# ========================

def target_function(X):
    return (1.0 
            + 0.5 * X[:,0] 
            + 2.1 * X[:,1] 
            + 1.8 * X[:,2] 
            + 0.7 * X[:,3] 
            + 3.3 * np.tanh(X[:,4]))

np.random.seed(1)
X = np.random.randn(1000, 5)
y_true = target_function(X)
y = y_true + np.random.normal(0, 0.5, size=y_true.shape)

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

X_train_bias = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
X_test_bias = np.hstack([np.ones((X_test.shape[0], 1)), X_test])

# Функция матричного умножения "с нуля"
def matmul(A, B):
    res = [[0]*len(B[0]) for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                res[i][j] += A[i][k] * B[k][j]
    return np.array(res)

# Транспонирование матрицы "с нуля"
def transpose(A):
    return np.array([[A[j][i] for j in range(len(A))] for i in range(len(A[0]))])

# Обращение матрицы методом Гаусса-Жордана
def invert_matrix(A):
    n = len(A)
    A = np.array(A, dtype=float)
    I = np.eye(n)
    AI = np.hstack([A, I])
    for i in range(n):
        AI[i] = AI[i] / AI[i, i]
        for j in range(n):
            if i != j:
                AI[j] = AI[j] - AI[i] * AI[j, i]
    return AI[:, n:]

# ========================
# Метрики
# ========================

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

# ========================
# 2. Методы оптимизации
# ========================

# ======== МНК ========
def MNK(X, y):
    XT = transpose(X)
    XTX = matmul(XT, X)
    XTy = matmul(XT, y.reshape(-1,1))
    XTX_inv = invert_matrix(XTX)
    theta = matmul(XTX_inv, XTy)
    return theta.flatten()

# ======== GD ========
def gradient_descent_full(X_train, y_train, X_test, y_test, lr=0.01, epochs=200):
    m, n = X_train.shape
    theta = np.zeros(n)
    history_train = []
    history_test = []
    
    for _ in range(epochs):
        preds_train = X_train @ theta
        error = preds_train - y_train
        grad = (1/m) * (X_train.T @ error)
        theta -= lr * grad
        
        history_train.append(mse(y_train, preds_train))
        history_test.append(mse(y_test, X_test @ theta))
        
    return theta, history_train, history_test

# ======== SGD ========
def sgd_full(X_train, y_train, X_test, y_test, lr=0.01, epochs=20, batch_size=32):
    m, n = X_train.shape
    theta = np.zeros(n)
    history_train = []
    history_test = []
    
    for _ in range(epochs):
        indices = np.random.permutation(m)
        for start in range(0, m, batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]
            xi_batch = X_train[batch_indices]
            yi_batch = y_train[batch_indices]
            grad = xi_batch.T @ (xi_batch @ theta - yi_batch) / len(batch_indices)
            theta -= lr * grad
            
        history_train.append(mse(y_train, X_train @ theta))
        history_test.append(mse(y_test, X_test @ theta))
        
    return theta, history_train, history_test

# ======== Adamax ========
def adamax(X_train, y_train, X_test, y_test, lr=0.002, epochs=200, beta1=0.9, beta2=0.999, eps=1e-8):
    m, n = X_train.shape
    theta = np.zeros(n)
    m_t = np.zeros(n)
    u_t = np.zeros(n)
    history_train = []
    history_test = []

    for t in range(1, epochs + 1):
        preds = X_train @ theta
        error = preds - y_train
        grad = (1/m) * (X_train.T @ error)

        m_t = beta1 * m_t + (1 - beta1) * grad
        u_t = np.maximum(beta2 * u_t, np.abs(grad))

        m_hat = m_t / (1 - beta1**t)
        theta -= (lr / (u_t + eps)) * m_hat

        history_train.append(mse(y_train, X_train @ theta))
        history_test.append(mse(y_test, X_test @ theta))

    return theta, history_train, history_test

# ========================
# Запуск методов и измерение времени
# ========================

# МНК
start = time.time()
theta_exact = MNK(X_train_bias, y_train)
end = time.time()
mse_train_exact = mse(y_train, X_train_bias @ theta_exact)
mse_test_exact = mse(y_test, X_test_bias @ theta_exact)
time_exact = end - start
iters_exact = 1

# GD
start = time.time()
theta_gd, hist_train_gd, hist_test_gd = gradient_descent_full(
    X_train_bias, y_train, X_test_bias, y_test, lr=0.01, epochs=200
)
end = time.time()
mse_train_gd = hist_train_gd[-1]
mse_test_gd = hist_test_gd[-1]
time_gd = end - start
iters_gd = 200

# SGD
start = time.time()
theta_sgd, hist_train_sgd, hist_test_sgd = sgd_full(
    X_train_bias, y_train, X_test_bias, y_test, lr=0.01, epochs=200, batch_size=32
)
end = time.time()
mse_train_sgd = hist_train_sgd[-1]
mse_test_sgd = hist_test_sgd[-1]
time_sgd = end - start
iters_sgd = 200

# Adamax
start = time.time()
theta_adamax, hist_train_adamax, hist_test_adamax = adamax(
    X_train_bias, y_train, X_test_bias, y_test, lr=0.01, epochs=200
)
end = time.time()
mse_train_adamax = hist_train_adamax[-1]
mse_test_adamax = hist_test_adamax[-1]
time_adamax = end - start
iters_adamax = 200

# ========================
# Финальная таблица сравнения
# ========================

df = pd.DataFrame({
    "Метод": ["МНК", "GD", "SGD", "Adamax"],
    "MSE train": [mse_train_exact, mse_train_gd, mse_train_sgd, mse_train_adamax],
    "MSE test": [mse_test_exact, mse_test_gd, mse_test_sgd, mse_test_adamax],
    "Количество итераций": [iters_exact, iters_gd, iters_sgd, iters_adamax],
    "Время выполнения (сек)": [time_exact, time_gd, time_sgd, time_adamax]
})

print(df)

# ========================
# Визуализация MSE на обучении
# ========================

plt.figure(figsize=(10,6))
plt.plot(hist_train_gd, label="GD")
plt.plot([i * (len(hist_train_gd)//len(hist_train_sgd)) for i in range(len(hist_train_sgd))], hist_train_sgd, label="SGD")
plt.plot(hist_train_adamax, label="Adamax", linewidth=2)
plt.axhline(y=mse_train_exact, color='r', linestyle='--', label="Точное решение (МНК)")
plt.xlabel("Эпоха")
plt.ylabel("MSE на обучающей выборке")
plt.title("Сравнение методов оптимизации (train)")
plt.legend()
plt.grid(True)
plt.show()

# ========================
# Визуализация MSE на тестовой выборке
# ========================
plt.figure(figsize=(10,6))
plt.plot(hist_test_gd, label="GD")
plt.plot([i * (len(hist_test_gd)//len(hist_test_sgd)) for i in range(len(hist_test_sgd))], hist_test_sgd, label="SGD")
plt.plot(hist_test_adamax, label="Adamax", linewidth=2)
plt.axhline(y=mse_test_exact, color='r', linestyle='--', label="Точное решение (МНК)")
plt.xlabel("Эпоха")
plt.ylabel("MSE на тестовой выборке")
plt.title("Сравнение методов оптимизации (test)")
plt.legend()
plt.grid(True)
plt.show()

# ========================
# Визуализация предсказаний для первых 100 точек
# ========================
n_points = 100
y_true_subset = y_train[:n_points]
X_subset_bias = X_train_bias[:n_points]

y_pred_exact = X_subset_bias @ theta_exact
y_pred_gd = X_subset_bias @ theta_gd
y_pred_sgd = X_subset_bias @ theta_sgd
y_pred_adamax = X_subset_bias @ theta_adamax

plt.figure(figsize=(10,6))
plt.plot(y_true_subset, label="Истинная целевая функция", linewidth=2)
plt.plot(y_pred_exact, '--', label="Предсказания МНК", linewidth=2)
plt.plot(y_pred_gd, ':', label="Предсказания GD", linewidth=2)
plt.plot(y_pred_sgd, '-.', label="Предсказания SGD", linewidth=2)
plt.plot(y_pred_adamax, '-', label="Предсказания Adamax", linewidth=2)
plt.xlabel("Индекс примера")
plt.ylabel("y")
plt.title("Сравнение предсказаний с истинной функцией (100 точек)")
plt.legend()
plt.grid(True)
plt.show()

