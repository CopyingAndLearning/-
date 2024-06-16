"""
name：01、自学代码 & 2.1.1 test
time：2024/6/16 16:11
author：yxy
content：
"""

import numpy as np

# 自变量 x，可能是一个二维数组
x = np.array([[1], [2], [3]])  # 这个的意思就是，一个自变量决定了一个参数
print(x.shape)  # 3*1

# 实际的因变量值 y
y = np.array([2, 4, 6])  # 1 * 3

# 模型预测的因变量值 y_pred
y_pred = np.array([1, 2, 3])
# 计算点积
gradient = np.dot(x.T, (y - y_pred))  # 点积。
# print(gradient.shape)
print(gradient)
