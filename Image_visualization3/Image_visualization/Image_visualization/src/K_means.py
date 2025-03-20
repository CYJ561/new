"""
kmeans算法又名k均值算法,K-means算法中的k表示的是聚类为k个簇，means代表取每一个聚类中数据值的均值作为该簇的中心，或者称为质心，即用每一个的类的质心对该簇进行描述。
        其算法思想大致为：先从样本集中随机选取 k个样本作为簇中心，并计算所有样本与这 k个“簇中心”的距离，对于每一个样本，将其划分到与其距离最近的“簇中心”所在的簇中，对于新的簇计算各个簇的新的“簇中心”。
        根据以上描述，我们大致可以猜测到实现kmeans算法的主要四点：
                （1）簇个数 k 的选择
               （2）各个样本点到“簇中心”的距离
                （3）根据新划分的簇，更新“簇中心”
                （4）重复上述2、3过程，直至"簇中心"没有移动

"""
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import Hubert

# 计算欧拉距离
def calcDis(dataSet, centroids, k):
    clalist = []
    for data in dataSet:
        diff = np.tile(data, (k,
                              1)) - centroids  # 相减   (np.tile(a,(2,1))就是把a先沿x轴复制1倍，即没有复制，仍然是 [0,1,2]。 再把结果沿y方向复制2倍得到array([[0,1,2],[0,1,2]]))
        squaredDiff = diff ** 2  # 平方
        squaredDist = np.sum(squaredDiff, axis=1)  # 和  (axis=1表示行)
        distance = squaredDist ** 0.5  # 开根号
        clalist.append(distance)
    clalist = np.array(clalist)  # 返回一个每个点到质点的距离len(dateSet)*k的数组
    return clalist


# 计算质心
def classify(dataSet, centroids, k):
    # 计算样本到质心的距离
    clalist = calcDis(dataSet, centroids, k)
    # 分组并计算新的质心
    minDistIndices = np.argmin(clalist, axis=1)  # axis=1 表示求出每行的最小值的下标
    newCentroids = pd.DataFrame(dataSet).groupby(
        minDistIndices).mean()  # DataFramte(dataSet)对DataSet分组，groupby(min)按照min进行统计分类，mean()对分类结果求均值
    newCentroids = newCentroids.values

    # 计算变化量
    changed = newCentroids - centroids

    return changed, newCentroids


# 使用k-means分类
def kmeans_classify(dataSet, k):
    # 随机取质心
    centroids = random.sample(list(dataSet), k)  # 返回包含列表中任意k个项目的列表：
    # print(centroids)
    # 更新质心 直到变化量全为0
    changed, newCentroids = classify(dataSet, centroids, k)
    while np.any(changed != 0):
        changed, newCentroids = classify(dataSet, newCentroids, k)

    centroids = sorted(newCentroids.tolist())  # tolist()将矩阵转换成列表 sorted()排序

    # 根据质心计算每个集群
    cluster = []
    clalist = calcDis(dataSet, centroids, k)  # 调用欧拉距离
    min = np.argmin(clalist, axis=0)
    # print(len(min))
    cluster_index = []  # 集群元素坐标的索引号
    minDistIndices = np.argmin(clalist, axis=1)  # 返回每行或每列的最小值所在下标索引(每行)
    # print(minDistIndices)
    center = []
    for m in range(len(min)):
        center.append(dataSet[min[m]])
    for i in range(k):
        cluster.append([])
        cluster_index.append([])
    for i, j in enumerate(minDistIndices):  # enymerate()可同时遍历索引和遍历元素
        cluster[j].append(dataSet[i])
        cluster_index[j].append(i)
    # print(cluster_index)
    return center, cluster, cluster_index


# 创建数据集
# def createDataSet():
#
#     return [[1, 2], [1, 1], [6, 4], [2, 1], [6, 3], [5, 4]]
def createDataSet():
    x = np.array([[1, 2], [1, 1], [6, 4], [2, 1], [6, 3], [5, 4]])
    return x


if __name__ == '__main__':
    dataset = createDataSet()
    center, cluster, cluster_index = kmeans_classify(dataset, 2)
    print('质心为：%s' % center)
    print('集群为：%s' % cluster)
    print('簇：%s' % cluster_index)
    # Co = Hubert.Hubert_cluster(dataset,center,cluster_index,2)
    # print("Co:",Co)
    data = dataset.copy()
    delt = []
    for index in center:  # del center
        print(index)
        for c in range(len(data)):
            if (index == data[c]).all():
                 delt.append(c)
    data = np.delete(data,delt,0)
    # print(data)
    for i in range(len(data)):
        # if np.where(dataset ==)
        plt.scatter(data[i][0], data[i][1], marker='o', color='green', s=40, label='原始点')
        #  记号形状       颜色      点的大小      设置标签
    for j in range(len(center)):
        plt.scatter(center[j][0], center[j][1], marker='x', color='red', s=50, label='质心')
    plt.show()
