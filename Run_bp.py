import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
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
        x = self.output_layer(x)
        return x

class RegressionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def main():
    file_path = input("请输入xlsx或csv文件路径：")
    file_ext = os.path.splitext(file_path)[-1].lower()

    if file_ext == '.xlsx':
        data = pd.read_excel(file_path)
    elif file_ext == '.csv':
        data = pd.read_csv(file_path)
    else:
        raise ValueError("文件格式不支持，仅支持xlsx和csv")
    print("\n*****bp神经网络回归模型*****")

    columns = data.columns.tolist()
    target_column = columns[-1]
    feature_columns = columns[:-1]

    has_target = ~pd.isnull(data[target_column])
    train_val_data = data[has_target].copy()
    test_data = data[~has_target].copy()

    X_train_val = train_val_data[feature_columns].values
    y_train_val = train_val_data[target_column].values
    X_test = test_data[feature_columns].values

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    batch_size = 64
    train_dataset = RegressionDataset(X_train_scaled, y_train)
    val_dataset = RegressionDataset(X_val_scaled, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    ## 修改你希望的参数
    input_dim = X_train_scaled.shape[1]
    layer_sizes = [128, 64, 32]
    dropout_rate = 0.3
    learning_rate = 0.001
    epochs = 300

    model = BPNet(input_dim, layer_sizes, dropout_rate)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X, y in train_loader:
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                outputs = model(X)
                loss = criterion(outputs, y)
                val_loss += loss.item()

        print(
            f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss / len(train_loader):.4f}, Val Loss: {val_loss / len(val_loader):.4f}")

    # 计算训练集和验证集的MSE
    model.eval()
    train_preds, train_targets = [], []
    with torch.no_grad():
        for X, y in train_loader:
            outputs = model(X)
            train_preds.extend(outputs.numpy().flatten())
            train_targets.extend(y.numpy().flatten())
    train_mse = mean_squared_error(train_targets, train_preds)
    print(f"\n训练集MSE: {train_mse:.4f}")

    val_preds, val_targets = [], []
    with torch.no_grad():
        for X, y in val_loader:
            outputs = model(X)
            val_preds.extend(outputs.numpy().flatten())
            val_targets.extend(y.numpy().flatten())
    val_mse = mean_squared_error(val_targets, val_preds)
    print(f"验证集MSE: {val_mse:.4f}")

    # 测试集预测
    test_dataset = RegressionDataset(X_test_scaled, np.zeros(X_test.shape[0]))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    predictions = []
    with torch.no_grad():
        for X, _ in test_loader:
            outputs = model(X)
            predictions.extend(outputs.numpy())

    output_dir = os.path.dirname(file_path)
    output_file = os.path.join(output_dir, "res_BP.xlsx")

    results_df = test_data.copy()
    results_df[target_column] = np.squeeze(predictions)
    results_df.to_excel(output_file, index=False)

    print(f"预测结果已保存到: {output_file}")

if __name__ == "__main__":
    main()