"""
name：01、自学代码 & main
time：2024/6/16 15:55
author：yxy
content：线性回归源码
"""

import numpy as np

# 假设我们有以下数据
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)  # 自变量，需要是二维数组
y = np.array([2, 4, 5, 4, 5])  # 因变量

# 初始化参数
w = 0.0
b = 0.0

# 学习率和迭代次数
learning_rate = 0.01
iterations = 1000

# 手写梯度下降
# 梯度下降
for _ in range(iterations):
    # 预测值
    # 得到的参数
    y_pred = np.dot(X, w) + b

    # 损失函数的梯度
    dw = -(2 / len(X)) * np.dot(X.T, (y - y_pred))  # np.dot(X.T, (y - y_pred)) 梯度的向量表示
    db = -(2 / len(X)) * np.sum(y - y_pred)  # 均方误差

    # 更新参数
    w -= learning_rate * dw  # 这个是跟新梯度。梯度 * 学习率。
    b -= learning_rate * db

# 打印结果
# print(f"斜率（w）: {w}, 截距（b）: {b}\n")
