import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import acorr_ljungbox
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
import warnings
from pathlib import Path
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Font

warnings.filterwarnings("ignore")

plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def main():
    # 用户输入数据路径
    file_path = input("请输入数据文件路径（支持CSV和Excel文件）：")

    # 读取数据
    file_ext = Path(file_path).suffix.lower()
    if file_ext == '.csv':
        df = pd.read_csv(file_path)
    elif file_ext in ('.xls', '.xlsx'):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("不支持的文件类型，请使用CSV或Excel文件")

    # 自动选择最后一列并处理缺失值
    time_series = df.iloc[:, -1]
    valid_series = time_series.dropna()
    pre_num = len(time_series) - len(valid_series)

    if pre_num <= 0:
        print("最后一列数据完整，无需预测。")
        return

    # 数据检验
    print("\n=== 数据检验 ===")
    lb_p_value = test_white_noise(valid_series)
    print(f"\n白噪声检验的p值：{lb_p_value:.4f}")
    if lb_p_value > 0.05:
        print("数据是白噪声，无法使用ARIMA模型进行预测。")
        return

    # 自动选择ARIMA参数
    print("\n=== 模型参数选择 ===")
    model_order = find_arima_order(valid_series)
    p, d, q = model_order
    print(f"\n最优的ARIMA模型参数为：(p={p}, d={d}, q={q})")

    # 模型训练与预测
    model = ARIMA(valid_series, order=(p, d, q))
    model_fit = model.fit()

    # 计算并输出MSE
    fitted_values = model_fit.fittedvalues
    original_aligned = valid_series.loc[fitted_values.index]
    mse = ((original_aligned - fitted_values) ** 2).mean()
    print(f"\n原始数据与拟合数据的MSE：{mse:.4f}")

    # 输出模型详情
    print("\n=== 模型参数详情 ===")
    print(model_fit.summary())

    # 生成预测结果
    forecast = model_fit.forecast(steps=pre_num)
    last_index = valid_series.index[-1]
    forecast_index = pd.RangeIndex(start=last_index + 1, stop=last_index + pre_num + 1)
    forecast_series = pd.Series(forecast, index=forecast_index)

    # 保存预测结果
    # 保存预测结果
    output_path = Path(file_path).parent / "res_ARIMA.xlsx"
    # 将预测结果转换为DataFrame，列名为pre_ARIMA，并去除索引
    forecast_df = pd.DataFrame({'pre_ARIMA': forecast_series.values})
    forecast_df.to_excel(output_path, index=False)

    print(f"预测结果已保存至：{output_path}")

def test_white_noise(series, lags=10):
    """白噪声检验（Ljung-Box检验）"""
    Q = acorr_ljungbox(series, lags=[lags], return_df=True)
    return Q['lb_pvalue'].iloc[0]


def find_arima_order(series):
    """自动选择ARIMA参数"""
    model = auto_arima(
        series,
        seasonal=False,
        d=None,
        trace=True,
        error_action='ignore',
        suppress_warnings=True
    )
    return model.order


if __name__ == "__main__":
    main()