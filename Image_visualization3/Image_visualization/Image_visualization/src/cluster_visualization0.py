import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# 生成示例数据
X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# 使用 KMeans 进行聚类
kmeans = KMeans(n_clusters=4, random_state=0).fit(X)
centers = kmeans.cluster_centers_
labels = kmeans.labels_

# 为每个聚类分配颜色
colors = plt.cm.nipy_spectral(labels.astype(float) / len(np.unique(labels)))

# 创建图形和坐标轴
fig, ax = plt.subplots(figsize=(10, 8))

# 绘制数据点
scatter = ax.scatter(X[:, 0], X[:, 1], c=colors, s=40, label='数据点')

# 绘制聚类中心
center_scatter = ax.scatter(centers[:, 0], centers[:, 1],
                            marker='X', c='#FF5252', s=200, linewidths=3, label='聚类中心')

# 设置初始透明度
original_alpha = 0.5
scatter.set_alpha(original_alpha)

# 提前计算每个簇的数据点索引
cluster_indices = [np.where(labels == i)[0] for i in range(len(centers))]

# 预先创建透明度数组
alphas = np.full(len(labels), original_alpha)

# 定义鼠标移动事件处理函数
last_index = None
def on_motion(event):
    global last_index
    if event.inaxes == ax:
        # 检查鼠标是否在簇中心上
        cont, ind = center_scatter.contains(event)
        current_index = None
        if cont:
            # 获取当前鼠标所在的簇中心的索引
            current_index = ind['ind'][0]
            if current_index != last_index:
                # 高亮显示该簇的数据点
                alphas.fill(0.1)
                alphas[cluster_indices[current_index]] = 1.0
                scatter.set_alpha(alphas)
                fig.canvas.draw_idle()
        else:
            if last_index is not None:
                # 如果鼠标不在簇中心上，恢复所有数据点的原始透明度
                alphas.fill(original_alpha)
                scatter.set_alpha(alphas)
                fig.canvas.draw_idle()
        last_index = current_index

# 连接鼠标移动事件
fig.canvas.mpl_connect('motion_notify_event', on_motion)

# 设置图例和标签
ax.legend()
ax.set_xlabel("UMAP 维度1")
ax.set_ylabel("UMAP 维度2")
ax.set_title("降维聚类结果")

# 显示图形
plt.show()    