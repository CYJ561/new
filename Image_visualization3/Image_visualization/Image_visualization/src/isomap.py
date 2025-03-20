"""
numpy.argpartition(a, kth, axis=-1, kind=‘introselect’, order=None)
（一）一般numpy中以arg开头的函数都是返回下标，而不改变原数组。

参数a是指传入的Numpy.array
参数kth是指列表中下标为k位置应该放置该数组中第k大的值

"""


import numpy as np
import matplotlib.pyplot as plt
import sys
from TSNE import tsne


# ISOMAP算法第一步骤
# 返回距离矩阵
def distancematrix(test):
    length = len(test)   # 获得矩阵行数
    resmat = np.zeros([length, length], np.float32)  # 构建length*length的方阵（0填充）,类型是float32
    # 返回给定形状和类型的新数组，用0填充
    for i in range(length):
        for j in range(length):
            # np.linalg.norm()用于求范数
            # np.linalg.norm(x, ord=None, axis=None, keepdims=False)
            # x表示矩阵，ord表示范数类型
            # ord=1：表示求列和的最大值
            # ord=2：|λE-ATA|=0，求特征值，然后求最大特征值得算术平方根
            # ord=∞：表示求行和的最大值
            # ord=None：表示求整体的矩阵元素平方和，再开根号
            resmat[i, j] = np.linalg.norm(test[i] - test[j])  # 记录i行和j行之间矩阵的距离，整体的矩阵元素平方和，再开根号
    # print("resmat,距离矩阵是:", resmat)
    return resmat  # 返回距离矩阵


# def mds(test, deg):
#     length = len(test)
#     re = np.zeros((length, length), np.float32)
#     if deg > length:
#         deg = length
#     D = distancematrix(test)
#     ss = 1.0 / length ** 2 * np.sum(D ** 2)
#     for i in range(length):
#         for j in range(length):
#             re[i, j] = -0.5 * (D[i, j] ** 2 - 1.0 / length * np.dot(D[i, :], D[i, :]) - 1.0 / length * np.dot(D[:, j], D[:,j]) + ss)
#
#     A, V = np.linalg.eig(re)
#     list_idx = np.argpartition(A, deg - 1)[-deg:]
#     a = np.diag(np.maximum(A[list_idx], 0.0))
#     return np.matmul(V[:, list_idx], np.sqrt(a))


# 使用 Dijkstra 算法获取最短路径，并更新距离矩阵
# test: 距离矩阵，大小 m * m
# start：最短路径的起始点，范围 0 到 m-1

# dij距离
def usedijk(test, start):
    count = len(test)
    col = test[start].copy()
    rem = count - 1
    while rem > 0:
        i = np.argpartition(col, 1)[1]
        length = test[start][i]
        for j in range(count):
            if test[start][j] > length + test[i][j]:
                test[start][j] = length + test[i][j]
                test[j][start] = test[start][j]
        rem -= 1
        col[i] = float('inf')


# test：需要降维的矩阵
# target：目标维度
# k：k 近邻算法中的超参数
# return：降维后的矩阵
def isomap(test, target, k):
    inf = float('inf')  # 正无穷
    count = len(test)  # 返回行的数量
    if k >= count:  # k >= 维度
        raise ValueError('K is too large')
    mat_distance = distancematrix(test)  # 返回距离矩阵
    # print("distance是距离矩阵")

    # 返回给定形状和数据类型的新数组，其中元素的值设置为1
    # np.ones(shape, dtype=None, order='C')
    # 1.shape：一个整数类型或者一个整数元组，用于定义数组的大小。如果仅指定一个整数类型变量，则返回一维数组。如果指定的是整数元组，则返回给定形状的数组。
    # 2.dtype：可选参数，默认值为float。用于指定数组的数据类型。
    # 3.order：指定内存重以行优先(‘C’)还是列优先(‘F’)顺序存储多维数组。
    knear = np.ones([count, count], np.float32) * inf  # 创建一个count*count的方阵，内容为inf（无穷大）
    for idx in range(count):   # 遍历行数
        topk = np.argpartition(mat_distance[idx], k)[:k + 1]  # 保证第k位的正确，使小于第k的数置于前，大的数置于后面，取出小于第k的数据
        # print("topK", topk)
        # np.argpartition（）用于返回按k分割后的顺序下标数组
        # 索引 0至k行
        knear[idx][topk] = mat_distance[idx][topk]

    for idx in range(count):
        usedijk(knear, idx)
    # return mds(knear, target)
    return knear


if __name__ == '__main__':
    print('开始降维.....')
    D = np.array([[1, 2, 3, 4], [2, 1, 5, 6], [3, 5, 1, 7], [4, 6, 7, 1]])  # test data
    print(D)
    outcome = isomap(D, 2, 3)
    sys.stdout.write('降维完成\n')
    print(outcome)


