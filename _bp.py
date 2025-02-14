import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np


class BPNet(nn.Module):
    def __init__(self, input_dim, layer_sizes, dropout_rate):
        super(BPNet, self).__init__()
        self.hidden_layers = nn.Sequential()
        prev_size = input_dim

        for i, size in enumerate(layer_sizes):
            self.hidden_layers.add_module(f"fc{i + 1}", nn.Linear(prev_size, size))
            self.hidden_layers.add_module(f"relu{i + 1}", nn.ReLU())
            self.hidden_layers.add_module(f"dropout{i + 1}", nn.Dropout(dropout_rate))
            prev_size = size

        self.output_layer = nn.Linear(prev_size, 1)

    def forward(self, x):
        x = self.hidden_layers(x)
        return self.output_layer(x)


class RegressionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx] if self.y is not None else self.X[idx]


def run_bp(file_path):
    # 读取数据
    if file_path.endswith('.xlsx'):
        data = pd.read_excel(file_path)
    else:
        data = pd.read_csv(file_path)

    # 数据预处理
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    # 分离待预测数据
    has_target = ~np.isnan(y)
    if np.sum(~has_target) == 0:
        return np.array([]), 0.0

    X_train_val = X[has_target]
    y_train_val = y[has_target]
    X_test = X[~has_target]

    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_val)
    X_test_scaled = scaler.transform(X_test)

    # 训练验证集分割
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_scaled, y_train_val, test_size=0.2, random_state=42
    )

    # 创建数据加载器
    batch_size = 64
    train_loader = DataLoader(
        RegressionDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        RegressionDataset(X_val, y_val),
        batch_size=batch_size,
        shuffle=False
    )

    # 模型参数
    model = BPNet(
        input_dim=X_train.shape[1],
        layer_sizes=[128, 64, 32],
        dropout_rate=0.3
    )
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 300

    # 训练循环
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                val_loss += criterion(outputs, y_batch).item()

    # 计算训练集MSE
    model.eval()
    with torch.no_grad():
        train_preds = model(torch.tensor(X_train_scaled, dtype=torch.float32))
        mse = criterion(train_preds, torch.tensor(y_train_val, dtype=torch.float32).view(-1, 1)).item()

    # 预测测试集
    test_dataset = RegressionDataset(X_test_scaled, None)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    predictions = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            outputs = model(X_batch)
            predictions.extend(outputs.numpy().flatten())

    return np.array(predictions), mse