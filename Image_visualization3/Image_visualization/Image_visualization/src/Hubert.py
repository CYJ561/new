import numpy as np

# 计算距离矩阵
def distance_matrix(test):
    """
    计算输入矩阵中每行之间的欧几里得距离矩阵
    :param test: 输入矩阵
    :return: 距离矩阵
    """
    length = len(test)
    resmat = np.zeros([length, length], np.float32)
    for i in range(length):
        for j in range(length):
            resmat[i][j] = np.linalg.norm(test[i] - test[j])
    return resmat

# 计算簇的关系矩阵
def distance_cluster(x, center, num):
    """
    计算簇之间的关系矩阵
    :param x: 簇的分类索引号
    :param center: 簇的中心点
    :param num: 元素个数
    :return: 簇的关系矩阵
    """
    dist_k = distance_matrix(center)
    C_matrix = np.zeros((num, 2))
    M_matrix = np.zeros((num, num))
    for j in range(num):
        C_matrix[j][0] = j
    for cluster in range(len(x)):
        for i in x[cluster]:
            C_matrix[i][1] = cluster
    for x_idx in range(num):
        for y_idx in range(num):
            if C_matrix[x_idx][1] != C_matrix[y_idx][1]:
                M_matrix[x_idx][y_idx] = dist_k[int(C_matrix[x_idx][1])][int(C_matrix[y_idx][1])]
    return M_matrix

# 计算Hubert指数
def hubert_cluster(original_coordinates, center, cluster_index, k):
    """
    计算Hubert指数
    :param original_coordinates: 点的原坐标
    :param center: k个簇的中心点
    :param cluster_index: 点的索引值
    :param k: 参数
    :return: Hubert指数
    """
    try:
        dist_ij = distance_matrix(original_coordinates)
        num = original_coordinates.shape[0]
        dist_mij = distance_cluster(cluster_index, center, num)
        M = k * (k - 1) / 2
        sum_val = 0
        sum1 = 0
        sum2 = 0
        for i in range(num):
            for j in range(num):
                sum_val += dist_ij[i][j] * dist_mij[i][j]
                sum1 += dist_ij[i][j] * dist_ij[i][j]
                sum2 += dist_mij[i][j] * dist_mij[i][j]
        r = 1 / M * sum_val
        M_p = 1 / M * np.sum(dist_ij)
        M_c = 1 / M * np.sum(dist_mij)
        sigma_p = 1 / M * sum1 - M_p * M_p
        sigma_p = abs(sigma_p)
        sigma_c = 1 / M * sum2 - M_c * M_c
        sigma_c = abs(sigma_c)
        Co = abs(r - M_p * M_c) / ((sigma_p ** 0.5) * (sigma_c ** 0.5))
        return Co
    except Exception as e:
        print(f"计算Hubert指数时出错: {e}")
        return None