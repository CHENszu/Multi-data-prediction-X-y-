import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error

def run_Xgboost(file_path):
    data = pd.read_excel(file_path) if file_path.endswith('.xlsx') else pd.read_csv(file_path)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # 分离训练和测试数据
    missing_mask = pd.isnull(y)
    if missing_mask.sum() == 0:
        return np.array([]), 0.0

    X_train = X[~missing_mask]
    y_train = y[~missing_mask]
    X_test = X[missing_mask]

    # 训练模型
    model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    model.fit(X_train, y_train)

    # 计算MSE
    y_train_pred = model.predict(X_train)
    mse = mean_squared_error(y_train, y_train_pred)

    # 预测
    y_pred = model.predict(X_test)
    return y_pred, mse