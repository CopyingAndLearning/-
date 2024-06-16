"""
name：01、自学代码 & 1、sklearn的去除去空归一化
time：2024/6/16 14:47
author：yxy
content：
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler

# 假设 X 是你的原始数据，它是一个NumPy数组
X = np.array([[-10, 20], [30, -40], [-20, 10]])

# 初始化 MinMaxScaler，设置 feature_range 为 [1, 2] 以避免负数
scaler = MinMaxScaler(feature_range=(1, 2))

# 拟合数据并进行归一化
X_scaled = scaler.fit_transform(X)

# 打印归一化后的数据
print("Normalized data without negative values:")
print(X_scaled)
