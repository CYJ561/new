"""
np.square(x): 计算数组各元素的平方并输出数组
np.sum(a,axis)  求和函数
    a：要求和的数组
    axis：要求和数组的轴。
        默认情况下，axis = None 将对输入数组的所有元素求和。
        axis = 0,同一列元素相加后合并为一个一维数组
        axis = 1，同一行元素相加后合并为一个一维数组
    isinstance(object,classinfo)  用来判断一个函数是否是一个已知的类型
    object : 实例对象。
    classinfo : 可以是直接或者间接类名、基本类型或者由它们组成的元组。
    返回值：如果对象的类型与参数二的类型（classinfo）相同则返回 True，否则返回 False。
np.dot(x, y) ：用于向量点积和矩阵乘法
    x是m*n 矩阵 ，y是n*m矩阵
        如果处理的是一维数组，则得到的是两数组的內积。
        如果是二维数组（矩阵）之间的运算，则得到的是矩阵积

"""

import numpy as np
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import time
from sklearn.manifold import TSNE


# 返回任意两个点之间的距离  ||x_i-x_j||^2
def cal_pairwise_dist(x):
    """
    (a-b)^2 = a^2 + b^2 - 2*a*b
    """
    # 将每行元素平方后相加，合并为一个一维数组
    sum_x = np.sum(np.square(x), 1)
    # x.T是x的转置矩阵  (-2)x(xT) + sum_x
    dist = np.add(np.add(-2 * np.dot(x, x.T), sum_x).T, sum_x)
    # 返回任意两个点之间距离的平方
    return dist
    TSNE

# 计算P_ij
def cal_perplexity(dist, idx=0, beta=1.0):  # dist是第i个点与其他点之间的距离（行向量）
    """计算perplexity, D是距离向量，
    idx指dist中自己与自己距离的位置，beta是高斯分布参数
    beta = 1/2σ^2
    """
    prob = np.exp(-dist * beta)  # 计算P_ij公式的分子，即exp(-||x_i-x_j||^2)*beta
    # 设置自身prob为0
    prob[idx] = 0
    sum_prob = np.sum(prob)  # 求分母
    prob /= sum_prob  # 求P_ij

    return prob


# 二分搜索寻找最优的σ下的P_ij
def seach_prob(x, tol=1e-5, perplexity=30.0):  # 目标困惑值为30.0
    """
    二分搜索寻找beta,并计算pairwise的prob
    """

    # 初始化参数
    (n, d) = x.shape  # 获得行、列
    dist = cal_pairwise_dist(x)  # 计算x中任意两点距离
    dist[dist < 0] = 0
    pair_prob = np.zeros((n, n))   # 条件概率初始化
    beta = np.ones((n, 1))  # beta = 1/(2*σ^2）,初始化每对点的beta为1
    # 取log，方便后续计算
    base_perp = np.log(perplexity)  # 期望值

    # 对每个样本点搜索最优的σ、beta，并计算对应的P_ij
    for i in range(n):
        if i % 500 == 0:
            print("Computing pair_prob for point %s of %s ..." % (i, n))

        betamin = -np.inf  # 负无穷
        betamax = np.inf  # 正无穷
        this_prob = cal_perplexity(dist[i], i, beta[i])

        # 二分搜索,寻找最佳sigma下的prob

        tries = 0
        while np.abs(perp_diff) > tol and tries < 50:  # 轮次小于50次
            # 如果交叉熵比期望值大，减小熵，增大beta（熵和Δ成正比，和beta反比）
            if perp_diff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2  # 增大beta
                else:
                    beta[i] = (beta[i] + betamax) / 2  # 在beta--betmax之间寻找最优beta
            else:
                # 交叉熵比期望值小，增大熵，减小beta
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2  # 缩小beta
                else:
                    beta[i] = (beta[i] + betamin) / 2  # 在betamin--beta之间寻找最优beta

            # 重新计算，更新perb,prob值
            perp, this_prob = cal_perplexity(dist[i], i, beta[i])
            perp_diff = perp - base_perp
            tries = tries + 1  # 轮次+1
        # 记录prob值
        pair_prob[i, ] = this_prob
    print("Mean value of sigma: ", np.mean(np.sqrt(1 / beta)))
    # 每个点对其他点的条件概率分布pi\j
    return pair_prob


def tsne(x, no_dims=2, perplexity=30.0, max_iter=1000):
    """
    max-iter:迭代轮次
    perplexity：目标值5-50之间
    """

    # Check inputs
    if isinstance(no_dims, float):  # 判断no_dims和float类型是否一致，是则返回True
        print("Error: array x should have type float.")
        return -1

    (n, d) = x.shape  # 返回x的维度（个数，维度）

    # 随机初始化地位数据Y
    y = np.random.randn(n, no_dims)  # 随机数生成函数，生成一个（n,no_dims）的二维数组

    # 对称化
    P = seach_prob(x, 1e-5, perplexity)  # 二分查找最优beta下的P_ij
    # 计算联合概率
    P = P + np.transpose(P)  # P+P.T(转置矩阵)
    P = P / np.sum(P)   # 高维数据的联合概率pij

    # P_ij，提前夸大
    print("T-SNE DURING:%s" % time.perf_counter())
    # 开始若干轮对P进行放大
    P = P * 4
    P = np.maximum(P, 1e-12)

    # 开始迭代训练
    initial_momentum = 0.5  # 初始冲量值
    final_momentum = 0.8  # 最终冲量值
    eta = 500  # 学习率100-1000
    min_gain = 0.01  # 增益参数
    # 梯度
    dy = np.zeros((n, no_dims))  # 全0的一个（n,no_dims）的二维数组
    # Y的变化
    iy = np.zeros((n, no_dims))  # 全0的一个（n,no_dims）的二维数组
    # 增益点
    gains = np.ones((n, no_dims))  # 全1的一个（n,no_dims）的二维数组
    # 梯度下降
    for iter in range(max_iter):

        # 计算q_ij,记为Q
        sum_y = np.sum(np.square(y), 1)  # 先平方
        num = 1 / (1 + np.add(np.add(-2 * np.dot(y, y.T), sum_y).T, sum_y))  # 分子，求法同34行
        num[range(n), range(n)] = 0  # 由于分母规定k!=l,则设置i和i之间的距离等于0
        Q = num / np.sum(num)   # qij
        Q = np.maximum(Q, 1e-12)    # X与Y逐位比较取其大者

        # 计算梯度 = 4sum[(P_ij-q_ij) * (y_i-y_j) * (1+||y_i-Y-j||^2)^(-1)]
        PQ = P - Q
        # 梯度dy
        for i in range(n):
            # 第i+1行dy更改为对应梯度
            # np.tile对输入的数组，元组或列表进行重复构造，其输出是数组
            dy[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i], (no_dims, 1)).T * (y[i, :] - y), 0)

        # 开始梯度下降（带有冲量）
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum
        # 如果梯度的方向和变化方向相同，增益*0.8；若不相同，增益+0.2
        gains = (gains + 0.2) * ((dy > 0) != (iy > 0)) + (gains * 0.8) * ((dy > 0) == (iy > 0))
        gains[gains < min_gain] = min_gain
        # 迭代，保持上次梯度变化的趋势
        iy = momentum * iy - eta * (gains * dy)
        y = y + iy

        # Y去中心化
        # np.mean(y, 0)求取平均值（0表示计算每一列的均值），返回一个向量
        # np.tile（a,(n, 1)）将a沿y轴扩大的倍数为n，沿x轴扩大的倍数为1
        y = y - np.tile(np.mean(y, 0), (n, 1))  # 点的偏移对点点之间的距离无影响，作用是保证每次降维后的中心点在0，0位置

        # 损失函数的计算
        if (iter + 1) % 100 == 0:  # 每隔100步，计算C
            C = np.sum(P * np.log(P / Q))
            print("Iteration ", (iter + 1), ": error is ", C)  # 打印C
            if (iter+1) != 100:
                ratio = C/oldC
                print("ratio ", ratio)
                if ratio >= 0.95:
                    break
            oldC = C

        # 停止放大P，还原P
        if iter == 100:
            P = P / 4
    print("finished training!")
    return y


if __name__ == "__main__":
    digits = load_digits()  # # 加载手写数字数据集
    # X = digits.data  # 创建特征矩阵
    X = np.array([[1, 2, 3, 4, 2, 1, 5, 6, 3, 5, 1, 7, 4, 6, 7, 1,1, 2, 3, 4, 2, 1, 5, 6, 3, 5, 1, 7, 4, 6, 7, 1,1, 2, 3, 4, 2, 1, 5, 6, 3, 5, 1, 7, 4, 6, 7, 1,1, 2, 3, 4, 2, 1, 5, 6, 3, 5, 1, 7, 4, 6, 7, 1],[2, 1, 5, 6, 3, 5, 1, 7, 4, 6, 7, 1, 1, 2, 3, 4,2, 1, 5, 6, 3, 5, 1, 7, 4, 6, 7, 1, 1, 2, 3, 4,2, 1, 5, 6, 3, 5, 1, 7, 4, 6, 7, 1, 1, 2, 3, 4,2, 1, 5, 6, 3, 5, 1, 7, 4, 6, 7, 1, 1, 2, 3, 4]])
    D = np.array([[1, 2, 3, 4], [2, 1, 5, 6], [3, 5, 1, 7], [4, 6, 7, 1]])  # test data
    print("x是", X)
    Y = digits.target  # 创建目标向量
    print("y是", Y)
    data_2d = tsne(X, 2)
    plt.scatter(data_2d[:, 0], data_2d[:, 1])
    plt.show()
