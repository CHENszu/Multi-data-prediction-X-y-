import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

def run_Lasso(file_path):
    data = pd.read_excel(file_path) if file_path.endswith('.xlsx') else pd.read_csv(file_path)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    # 分离训练和测试数据
    nan_indices = np.isnan(y)
    if np.sum(nan_indices) == 0:
        return np.array([]), 0.0

    X_train = X[~nan_indices]
    y_train = y[~nan_indices]
    X_test = X[nan_indices]

    # 训练模型
    lasso = Lasso(alpha=1.0, max_iter=10000)
    lasso.fit(X_train, y_train)

    # 计算MSE
    y_pred = lasso.predict(X_train)
    mse = mean_squared_error(y_train, y_pred)

    # 预测
    y_test_pred = lasso.predict(X_test)
    return y_test_pred, mse