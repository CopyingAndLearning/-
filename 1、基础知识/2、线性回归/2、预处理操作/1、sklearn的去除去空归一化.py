"""
name：01、自学代码 & 1、sklearn的去除去空归一化
time：2024/6/16 14:47
author：yxy
content：
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 假设 X 是你的原始数据，它是一个NumPy数组
X = np.array([[-10, 20], [30, -40], [-20, 10]])
#
# # 初始化 MinMaxScaler，设置 feature_range 为 [1, 2] 以避免负数
# scaler = MinMaxScaler(feature_range=(1, 2))         # 可以规定对应的归一化的范围。这里是将值映射到1-2之间的。
#
# # 拟合数据并进行归一化
# X_scaled = scaler.fit_transform(X)
#
# # 打印归一化后的数据
# print("Normalized data without negative values:")
# print(X_scaled)
# print(...)


# ------------------------ #
# 绘制归一化，前后图形的变化 #
import matplotlib.pyplot as plt
import seaborn as sns

# 假设 df 是包含数值特征的 DataFrame
df = pd.DataFrame(X, columns=['c1', 'c2'])  # 将np.array数据转换为pd.df数据

# 初始化 MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))  # 缩放到一定比例区间

# 对所有数值列进行归一化
df_normalized = scaler.fit_transform(df)

# 绘制归一化前后的直方图
plt.figure(figsize=(10, 6))
for i in range(df.shape[1]):
    plt.subplot(df.shape[1], 2, 2 * i + 1)
    sns.histplot(df.iloc[:, i], kde=True)
    plt.title(f"Original - Column {i}")

    plt.subplot(df.shape[1], 2, 2 * i + 2)
    sns.histplot(df_normalized[:, i], kde=True)
    plt.title(f"Normalized - Column {i}")

plt.tight_layout()
plt.show()
