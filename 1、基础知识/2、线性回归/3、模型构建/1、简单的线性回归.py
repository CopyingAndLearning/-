"""
name：01、自学代码 & 1、简单的线性回归
time：2024/6/16 15:26
author：yxy
content：简单的线性回归
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# 假设我们有以下数据
X = np.array([1, 2, 3, 4, 5])  # 自变量，需要转换为二维数组 [[1], [2], ...]
y = np.array([2, 4, 5, 4, 5])  # 因变量
# 将X转换为列向量，添加偏置项
X = X.reshape(-1, 1)  # np一维数据升维的方法
X = np.hstack((np.ones((X.shape[0], 1)), X))  # 添加全1的列
# 创建线性回归模型实例
model = LinearRegression()

# 训练模型
model.fit(X, y)  # 模型拟合自变量和因变量。

# 使用模型进行预测
y_pred = model.predict(X)  # 预测。
print(y_pred)

# 评估模型性能
r2 = r2_score(y, y_pred)
print(f"模型的R²分数是: {r2}")

# 打印模型参数（斜率和截距）
print(f"斜率: {model.coef_[0]}")
print(f"截距: {model.intercept_}")
