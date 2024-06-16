"""
name：01、自学代码 & test
time：2024/6/11 21:27
author：yxy
content：
"""

# import numpy as np
# import matplotlib.pyplot as plt
#
# # 生成随机数据
# np.random.seed(0)  # 为了可重复性设置随机种子
# data = np.random.randn(100, 2)  # 生成100个观测，2个特征
#
# # 绘制散点图
# plt.scatter(data[:, 0], data[:, 1])
# plt.title('Random Data Scatter Plot')
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# plt.show()

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 生成随机数据
np.random.seed(0)  # 指定对应的种子，方便随机数据一致
# 注：random随机的数据都是小数级别的
data = np.random.randn(100, 3)  # 生成100个观测，3个特征 # 数据的形式为：[ 0.49374178 -0.11610394 -2.03068447]
# print(data)
# print(data[:, 1])               # 全部行都要，但是只要第2列数据。  # 0,1,2    # 其他高纬度数据同理。

# # 绘制原始数据的散点图
plt.figure(figsize=(12, 6))  # 指定宽高，前者为长

#
# # 原始数据
plt.subplot(1, 2, 1)  # 1行2列的第一个子图
# cmap表示在预定义的颜色图中，选择一个指定的颜色。
# s表示指定对应的点的大小，alpha指定图的点的透明度
plt.scatter(data[:, 0], data[:, 1], c=data[:, 2], cmap='viridis', s=50, alpha=0.5)
# 指定图像的名字
plt.title('Original Data')
# 指定x和y轴的名字
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

#
# 数据标准化
scaler = StandardScaler()  # 将数据进行标准化，在进行归一化。
# 将数据应用到归一化的模型当中
data_scaled = scaler.fit_transform(data)
#
# PCA降维
pca = PCA(n_components=2)  # 降维到2个主成分 # n_components将数据降到2维。
data_pca = pca.fit_transform(data_scaled)  # 拟合标准化后的数据

# # 降维后的数据
plt.subplot(1, 2, 2)  # 1行2列的第二个子图
# 降维的目的是为了更好的进行聚类分析
plt.scatter(data_pca[:, 0], data_pca[:, 1], c=data_pca[:, 1], cmap="viridis", s=50)  # 降维之后第三个特征就不见了
plt.title('PCA Reduced Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

# plt.tight_layout()    # 使对应的布局更加好看
plt.show()
