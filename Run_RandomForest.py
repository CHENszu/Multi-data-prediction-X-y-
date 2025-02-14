import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import os


# 获取用户输入的文件路径
def get_file_path():
    file_path = input("请输入xlsx或csv文件的路径：").strip()
    if not (file_path.endswith('.xlsx') or file_path.endswith('.csv')):
        raise ValueError("文件格式不匹配，请输入.xlsx或.csv文件的路径。")
    return file_path


# 读取数据
def read_data(file_path):
    if file_path.endswith('.xlsx'):
        data = pd.read_excel(file_path)
    elif file_path.endswith('.csv'):
        data = pd.read_csv(file_path)
    else:
        raise ValueError("不支持的文件格式，请检查输入路径.")
    return data


# 数据预处理
def preprocess_data(data):
    X = data.iloc[:, :-1]  # 自变量
    y = data.iloc[:, -1]  # 因变量

    # 找出包含缺失值的行（因变量最后缺失的部分）
    train_mask = y.notna()
    test_mask = y.isna()

    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[test_mask]

    return X_train, y_train, X_test, data


# 训练随机森林模型并预测
def train_and_predict(X_train, y_train, X_test):
    # 打印当前模型
    print("\n*****随机森林回归模型*****")

    # 初始化随机森林回归模型
    regressor = RandomForestRegressor(n_estimators=100, random_state=42)

    # 训练模型
    regressor.fit(X_train, y_train)

    # 预测
    y_pred = regressor.predict(X_test)

    return regressor, y_pred


# 保存预测结果
def save_results(file_path, y_pred, input_data, X_test):
    # 创建结果保存路径
    output_path = os.path.join(os.path.dirname(file_path), 'res_RandomForest.xlsx')

    # 提取需要预测的数据
    result = input_data.loc[input_data.iloc[:, -1].isna()].copy()
    result.iloc[:, -1] = y_pred  # 填充预测结果

    # 保存为Excel文件
    result.to_excel(output_path, index=False)
    print(f"预测结果已保存到：{output_path}")


# 主函数
def main():
    try:
        # 获取文件路径
        file_path = get_file_path()

        # 读取数据
        data = read_data(file_path)

        # 数据预处理
        X_train, y_train, X_test, input_data = preprocess_data(data)

        # 训练模型
        regressor, y_pred = train_and_predict(X_train, y_train, X_test)

        # 打印训练后的MSE
        y_train_pred = regressor.predict(X_train)
        mse = mean_squared_error(y_train, y_train_pred)
        print(f"\n训练后的MSE（均方误差）：{mse:.4f}")

        # 保存结果
        save_results(file_path, y_pred, input_data, X_test)

    except Exception as e:
        print(f"发生错误：{e}")


if __name__ == "__main__":
    main()