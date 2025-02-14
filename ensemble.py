import pandas as pd
import numpy as np
from pathlib import Path
from _ARIMA import run_ARIMA
from _Lasso import run_Lasso
from _RandomForest import run_RandomForest
from _Xgboost import run_Xgboost
from _bp import run_bp

MODEL_MAP = {
    1: ("ARIMA", run_ARIMA),
    2: ("Lasso", run_Lasso),
    3: ("RandomForest", run_RandomForest),
    4: ("Xgboost", run_Xgboost),
    5: ("BP", run_bp)
}

def main():
    # 用户输入
    file_path = input("请输入数据文件路径（xlsx或者csv）：")
    model_nums = input("请选择要组合的模型编号（1-5，逗号分隔）：\n1-ARIMA(随时间变化才用)\n2-Lasso\n3-Random Forest\n4-Xgboost\n5-BP Network\n")
    selected = [int(x.strip()) for x in model_nums.split(",")]

    # 运行选中的模型
    results = {}
    mses = {}
    for num in selected:
        name, func = MODEL_MAP[num]
        print(f"\n=== 正在运行{name}模型... ===")
        pred, mse = func(file_path)
        results[name] = pred
        mses[name] = mse

    # 计算权重
    weights = {name: 1/mse for name, mse in mses.items()}
    total = sum(weights.values())
    weights = {name: w/total for name, w in weights.items()}

    print("\n=== 模型权重 ===")
    for name, w in weights.items():
        print(f"{name}: {w:.4f}")

    # 计算加权平均
    final_pred = np.zeros_like(next(iter(results.values())))
    for name, pred in results.items():
        final_pred += pred * weights[name]

    # 保存结果
    output_df = pd.DataFrame()
    for name, pred in results.items():
        output_df[f'pre_{name}'] = pred
    output_df['pre_final'] = final_pred

    output_path = Path(file_path).parent / "res.xlsx"
    output_df.to_excel(output_path, index=False)
    print(f"\n结果已保存至：{output_path}")

if __name__ == "__main__":
    main()