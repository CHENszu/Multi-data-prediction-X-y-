import pandas as pd
import numpy as np
from statsmodels.stats.diagnostic import acorr_ljungbox
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

def test_white_noise(series, lags=10):
    Q = acorr_ljungbox(series, lags=[lags], return_df=True)
    return Q['lb_pvalue'].iloc[0]

def run_ARIMA(file_path):
    # 读取数据
    file_ext = Path(file_path).suffix.lower()
    if file_ext == '.csv':
        df = pd.read_csv(file_path)
    elif file_ext in ('.xls', '.xlsx'):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("不支持的文件类型")

    time_series = df.iloc[:, -1]
    valid_series = time_series.dropna()
    pre_num = len(time_series) - len(valid_series)

    if pre_num <= 0:
        return np.array([]), 0.0  # 无预测需求

    # 数据检验
    lb_p_value = test_white_noise(valid_series)
    if lb_p_value > 0.05:
        return np.array([]), 0.0  # 白噪声不预测

    # 模型训练
    model_order = auto_arima(valid_series, seasonal=False, d=None, suppress_warnings=True).order
    model = ARIMA(valid_series, order=model_order)
    model_fit = model.fit()

    # 计算MSE
    fitted_values = model_fit.fittedvalues
    original_aligned = valid_series.loc[fitted_values.index]
    mse = ((original_aligned - fitted_values) ** 2).mean()

    # 预测
    forecast = model_fit.forecast(steps=pre_num)
    return forecast.values, mse