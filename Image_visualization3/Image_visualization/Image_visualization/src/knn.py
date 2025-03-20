"""
# reps:(m,n), 其中m {m}m控制纵向重复，n {n}n控制横向重复

sum(axis=1)
axis是求和的方向，为0表示沿着行的方向运算，1表示沿着列，为负数表示末尾开始计算

2.np.sqrt()  返回平方根
"""


import numpy as np
import operator


def knnClassify(inputX, data, labels, k):
    # 1.计算测试数据与各训练数据之间的距离。
    dataSize = data.shape[0]  # hang

    # 构建一个与训练数据相匹配的矩阵并与训练数据想减，获得差
    x = np.tile(inputX, (dataSize, 1)) - data  # 横向重复1刺，纵向重复dataSize次
    # print(np.tile(inputX, (dataSize, 1)))
    # print(x)
    # 对矩阵元素平方
    xPositive = x ** 2  # **表示乘方，此处为x的2次方
    # print(xPositive)
    xDistances = xPositive.sum(axis=1)  # 沿着列的方向累加（横向）
    # axis是求和的方向，为0表示沿着行的方向运算，1表示沿着列，为负数表示末尾开始计算
    # print(xDistances)

    # 点距
    distances = np.sqrt(xDistances)  # 求平方根
    # print(distances)

    # 2.按照距离的大小进行排序。
    sortDisIndex = distances.argsort()
    print(sortDisIndex)
    # 3.选择其中距离最小的k个样本点。4.确定K个样本点所在类别的出现频率。
    classCount = {}  # 创建字典：label为键，频数为值
    for i in range(k):
        getLabel = labels[sortDisIndex[i]]
        classCount[getLabel] = classCount.get(getLabel, 0) + 1

    # 5.返回K个样本点中出现频率最高的类别作为最终的预测分类。
    sortClass = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    print(sortClass[0][0])

    return sortClass[0][0]


if __name__ == '__main__':
    # 测试数据
    inputX = np.array([2, 2.2, 1.9])
    # 训练数据
    data = np.array([[1, 0.9, 1], [0.8, 0.9, 0.7], [1.3, 1, 1.2], [1.2, 0.9, 1], [2, 2.2, 2.1], [2.3, 2.2, 2],
                     [2, 2.2, 1.9], [1.9, 2.2, 2.1], [3.1, 3.1, 3], [2.8, 2.9, 3.1], [2.9, 3, 3.2], [3.1, 3, 3.1]])
    # 训练数据标签
    labels = np.array([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])

    k = 4

    knnClassify(inputX, data, labels, k)
