import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def run_RandomForest(file_path):
    data = pd.read_excel(file_path) if file_path.endswith('.xlsx') else pd.read_csv(file_path)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # 分离训练和测试数据
    train_mask = y.notna()
    test_mask = y.isna()
    if test_mask.sum() == 0:
        return np.array([]), 0.0

    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[test_mask]

    # 训练模型
    regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    regressor.fit(X_train, y_train)

    # 计算MSE
    y_train_pred = regressor.predict(X_train)
    mse = mean_squared_error(y_train, y_train_pred)

    # 预测
    y_pred = regressor.predict(X_test)
    return y_pred, mse