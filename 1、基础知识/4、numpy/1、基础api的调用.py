"""
name：01、自学代码 & 1、基础api的调用
time：2024/6/16 15:46
author：yxy
content：
"""

import numpy as np

# 创建两个一维数组
array1 = np.array([1, 2, 3])
array2 = np.array([4, 5, 6])

# ------------------ #
### 堆叠
## 当然也可以进行高维度的水平或垂直堆叠
# 水平堆叠数组
# hstacked_array = np.hstack((array1, array2))    # 按行堆叠，按axis=0进行堆叠。
# print(hstacked_array)

# 垂直堆叠数组
# hstacked_array = np.vstack((array1, array2))    # 按行堆叠，按axis=1进行堆叠。
# print(hstacked_array)
