import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim

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

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test  = torch.tensor(X_test, dtype=torch.float32)
y_test  = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# ======================================
# Модель PyTorch без активаций
# ======================================
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(X.shape[1], 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 1)

    def forward(self, x):
        x = self.fc1(x)   # без активации
        x = self.fc2(x)
        x = self.fc3(x)
        return x

model = Net()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 100
batch_size = 32

train_losses = []

# ======================================
# Обучение
# ======================================
for epoch in range(epochs):
    permutation = torch.randperm(X_train.size()[0])

    epoch_loss = 0

    for i in range(0, X_train.size(0), batch_size):
        idx = permutation[i:i+batch_size]
        xb = X_train[idx]
        yb = y_train[idx]

        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    train_losses.append(epoch_loss)

    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}, loss={epoch_loss:.4f}")

# ======================================
# Оценка
# ======================================
model.eval()
y_pred = model(X_test).detach().numpy().ravel()
y_test_np = y_test.numpy().ravel()

print("\nPyTorch MSE:", mean_squared_error(y_test_np, y_pred))
print("PyTorch MAE:", mean_absolute_error(y_test_np, y_pred))
print("PyTorch R² :", r2_score(y_test_np, y_pred))

# ======================================
# График MSE по эпохам
# ======================================
plt.plot(train_losses)
plt.title("PyTorch Train MSE")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.grid(True)
plt.show()
