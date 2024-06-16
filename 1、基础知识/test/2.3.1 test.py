"""
name：01、自学代码 & 2.3.1 test
time：2024/6/16 15:43
author：yxy
content：
"""

import numpy as np

# 假设我们有以下数据
X = np.array([1, 2, 3, 4, 5])  # 自变量，需要转换为二维数组 [[1], [2], ...]
y = np.array([2, 4, 5, 4, 5])  # 因变量

# 将X转换为列向量，添加偏置项
# print(X.shape)
X = X.reshape(-1, 1)  # np一维数据升维的方法
# print(X.shape)

# 水平堆叠 # 堆叠成一个，高维的数组
print(X)
X = np.hstack((np.ones((X.shape[0], 1)), X))  # 添加全1的列
print(np.ones((X.shape[0], 1)))
print(X)
