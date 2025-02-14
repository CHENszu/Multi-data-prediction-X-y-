import os
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

def main():
    # 提示用户输入文件路径
    file_path = input("请输入xlsx或csv文件的路径: ")
    file_name, file_extension = os.path.splitext(file_path)

    # 根据文件后缀选择读取方式
    if file_extension == '.xlsx':
        data = pd.read_excel(file_path)
    elif file_extension == '.csv':
        data = pd.read_csv(file_path)
    else:
        print("不支持的文件格式，请输入xlsx或csv文件！")
        return
    print("\n*****Lasso回归模型*****")
    # 数据预处理
    # 第一列为标签，最后一列为因变量，中间为自变量
    features = data.columns[:-1]  # 获取自变量列名
    X = data[features].values
    y = data.iloc[:, -1].values

    # 分离出完整的因变量数据和需要预测的数据
    total_rows = len(y)
    nan_indices = np.isnan(y)
    y_train = y[~nan_indices]
    X_train = X[~nan_indices]
    X_test = X[nan_indices]

    # 自变量归一化
    # scaler = StandardScaler()
    # X_train_scaled = scaler.fit_transform(X_train)
    # X_test_scaled = scaler.transform(X_test)
    X_train_scaled = X_train
    X_test_scaled = X_test

    # Lasso回归模型训练
    lasso = Lasso(alpha=1.0, max_iter=10000)  # Lasso回归模型，alpha为正则化参数
    lasso.fit(X_train_scaled, y_train)

    # 模型参数摘要
    print("Lasso回归模型参数:")
    coefficients = lasso.coef_
    selected_features = [features[i] for i in range(len(features)) if coefficients[i] != 0]
    print(f"选定的自变量: {selected_features}")
    print(f"非零系数的自变量数目: {len(selected_features)}")
    print(f"回归系数（非零）: {coefficients[coefficients != 0]}")
    print(f"截距: {lasso.intercept_}")

    # 模型拟合性能指标
    y_pred = lasso.predict(X_train_scaled)
    mse = mean_squared_error(y_train, y_pred)
    print(f"训练集上的MSE: {mse}")

    # 预测缺失部分的因变量
    y_test_pred = lasso.predict(X_test_scaled)

    # 将预测结果填充到原始数据中（仅保留测试部分）
    results = data[nan_indices].copy()  # 只保留因变量为NaN的部分
    results.iloc[:, -1] = y_test_pred

    # 保存结果到Excel文件
    res_file_path = os.path.join(os.path.dirname(file_path), f"res_Lasso.xlsx")
    results.to_excel(res_file_path, index=False)
    print(f"预测结果已保存到: {res_file_path}")

if __name__ == "__main__":
    main()