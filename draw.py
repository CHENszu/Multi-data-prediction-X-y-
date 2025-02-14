import pandas as pd
import matplotlib.pyplot as plt

# 读取Excel文件
file_path = 'res.xlsx'  # 替换为你的文件路径
data = pd.read_excel(file_path)

# 提取最后一列的数据
last_column = data.iloc[:, -1]

# 获取需要绘制的列数（排除最后一列）
num_columns = len(data.columns) - 1

# 检查是否有足够的列进行对比
if num_columns < 1:
    raise ValueError("数据至少需要包含两列才能进行比较")

# 获取数据点的索引
num_samples = len(data)
x = list(range(1, num_samples + 1))  # 横坐标为数据点索引

# 根据列数动态调整画布大小（每个子图高度设为4）
plt.figure(figsize=(12, 4 * num_columns))

for i in range(num_columns):
    column_name = data.columns[i]

    # 创建子图（垂直排列）
    plt.subplot(num_columns, 1, i + 1)

    # 绘制当前列和最后一列
    plt.plot(x, data[column_name], label=column_name)
    plt.plot(x, last_column, label=last_column.name, linestyle='--')

    # 设置图表元素
    plt.title(f'{column_name} vs. {last_column.name}')
    plt.xlabel('Data Index')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)

# 调整子图间距
plt.tight_layout()
plt.show()