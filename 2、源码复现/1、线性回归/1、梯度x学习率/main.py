"""
name：01、自学代码 & main
time：2024/6/16 16:55
author：yxy
content：向量移动可视化
"""

import numpy as np
from matplotlib import pyplot as plt

# 创建3D图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 假设我们有一组三维向量
vectors_3d = np.array([
    [3, 1, 2],  # 原向量
    [6, 2, 4],  # 梯度 * 学习率 决定原向量的移动。   # 经过线性变换，则他们会在同一条直线上
    [7, 8, 9]
])

# 绘制起点
ax.scatter(0, 0, 0, c='red', marker='o')

# 绘制向量
for vec in vectors_3d:
    # 起点坐标和各个终点的坐标。
    # arrow_length_ratio 控制轴的最小刻度
    ax.quiver(0, 0, 0, vec[0], vec[1], vec[2], arrow_length_ratio=10)

plt.show()
