"""
name：01、自学代码 & 1、绘图
time：2024/6/11 21:36
author：yxy
content：
"""

import matplotlib.pyplot as plt


# 测试数组的取值
def getArray():
    test = [1, 2, 3, 4, 5, 6]
    print("原数组test为丶：", test)
    print("倒着取test数组：", test[::-1])


def drawLC():
    # 创建一个新的图形窗口
    plt.figure(figsize=(10, 5))  # figsize 参数用来设置图形的大小，单位是英寸
    # 1、绘制折线图
    plt.plot([1, 2, 3], [4, 5, 6])  # 绘制一个简单的折线图
    plt.show()  # 显示图形


if __name__ == '__main__':
    getArray()
    drawLC()
