import os
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def main():
    # 提示用户输入文件路径
    file_path = input("请输入文件路径：")
    file_extension = os.path.splitext(file_path)[1].lower()

    # 检查文件格式
    if file_extension not in ['.csv', '.xlsx']:
        print("不支持的文件格式，请提供 .csv 或 .xlsx 文件。")
        return

    # 读取数据
    if file_extension == '.csv':
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)

    print("\n*****xgboost回归模型*****")
    # 数据预处理：最后一列是因变量，前面的列是自变量
    X = df.iloc[:, :-1]  # 自变量
    y = df.iloc[:, -1]   # 因变量

    # 分割数据集：将因变量为缺失值的行作为测试数据
    # 假设因变量的最后一些数据是缺失的
    missing_mask = pd.isnull(y)
    X_train = X[~missing_mask]
    y_train = y[~missing_mask]
    X_test = X[missing_mask]

    # 如果没有测试数据，提示用户
    if X_test.empty:
        print("没有找到需要预测的数据。")
        return

    # 使用 XGBoost 回归模型
    model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

    # 拟合模型
    model.fit(X_train, y_train)

    # 计算并打印模型的均方误差（MSE）
    y_train_pred = model.predict(X_train)  # 在训练集上进行预测
    mse = mean_squared_error(y_train, y_train_pred)
    print(f"模型均方误差（MSE）：{mse}")

    # 打印模型的评估结果
    print("模型评估：")
    print(f"超参数：{model.get_params()}")

    # 预测测试数据
    y_pred = model.predict(X_test)

    # 保存预测结果到 Excel 文件
    # 确保保存路径不为空
    save_path = os.path.join(os.path.dirname(file_path), 'res_Xgboost.xlsx')
    if save_path:
        # 创建包含预测结果的 DataFrame
        result_df = X_test.copy()
        result_df['Prediction'] = y_pred

        # 保存结果
        result_df.to_excel(save_path, index=False)
        print(f"预测结果已保存到：{save_path}")
    else:
        print("未能保存预测结果，请检查路径。")

if __name__ == "__main__":
    main()