"""
name：01、自学代码 & 1、基础api的调用
time：2024/6/16 15:03
author：yxy
content：
"""

import pandas as pd

df = pd.DataFrame({
    'A': [True, False, True],
    'B': [False, False, True]
})

# 1、选择指定数据类型的列
print(df.select_dtypes(include=['bool']).columns)
