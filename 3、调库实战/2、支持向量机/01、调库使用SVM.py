"""
name：01、自学代码 & 01、调库使用SVM
time：2024/6/16 10:56
author：yxy
content：
"""

from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# 加载数据集，这里以鸢尾花数据集为例
iris = datasets.load_iris()
X = iris.data
y = iris.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 创建SVM分类器
svm_classifier = SVC(kernel='linear')  # 这里使用线性核

# 训练模型
svm_classifier.fit(X_train_scaled, y_train)

# 预测测试集
y_pred = svm_classifier.predict(X_test_scaled)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
