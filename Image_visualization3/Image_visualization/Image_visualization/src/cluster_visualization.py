import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from PIL import Image
import streamlit as st
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# 假设这里有一个函数可以从图片中提取特征，这里简单用随机数代替
def extract_features_from_images(image_paths):
    num_images = len(image_paths)
    num_features = 10  # 假设每个图片有10个特征
    return np.random.rand(num_images, num_features)

# 加载图片路径
source_folder = 'D:/Image_visualization3/Image_visualization/Image_visualization/Source'
image_paths = [os.path.join(source_folder, f) for f in os.listdir(source_folder) if f.endswith(('.jpg', '.png'))]

# 提取图片特征
features = extract_features_from_images(image_paths)

# 使用KMeans进行聚类
n_clusters = 5  # 假设分为5个簇
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
labels = kmeans.fit_predict(features)

# 使用t-SNE进行降维
tsne = TSNE(n_components=2, random_state=0)
reduced_features = tsne.fit_transform(features)

# 创建DataFrame用于绘图
df = pd.DataFrame({
    'x': reduced_features[:, 0],
    'y': reduced_features[:, 1],
    'cluster': labels
})

# Streamlit应用
st.title('聚类可视化')

# 创建Figure和Axes
fig, ax = plt.subplots(figsize=(10, 8))

# 绘制初始聚类分布图
palette = sns.color_palette("husl", n_clusters)
scatter = sns.scatterplot(
    x='x', y='y',
    hue='cluster', palette=palette,
    s=60, edgecolor='w', linewidth=0.5,
    data=df,
    ax=ax
)

# 标注簇中心
centers = kmeans.cluster_centers_
centers_reduced = tsne.fit_transform(centers)
for i, center in enumerate(centers_reduced):
    ax.scatter(center[0], center[1], s=300, marker='*',
               c='gold', edgecolor='k', linewidth=1)

ax.set_title(f"2D 聚类分布 (共{n_clusters}个簇)", fontsize=12)
ax.set_xlabel('Feature 1', fontsize=10)
ax.set_ylabel('Feature 2', fontsize=10)

# 调整图例位置
legend = ax.legend(title='簇编号', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
legend.get_title().set_fontsize(8)

# 添加评估指标文字说明
textstr = '\n'.join([
    '评估指标说明：',
    '- 轮廓系数: [-1,1]，值越大越好',
    '- CH指数: 值越大表示分离度越好',
    '- DB指数: 值越小表示聚类越紧凑',
    '- 簇内距离: 平均值越小越好',
    '- 簇间距离: 最小值越大越好'
])
ax.text(1.05, 0.02, textstr, transform=ax.transAxes,
        verticalalignment='bottom', fontsize=8)

# 保存初始图像
canvas = FigureCanvas(fig)
canvas.draw()
buf = canvas.buffer_rgba()
img = Image.frombuffer('RGBA', canvas.get_width_height(), buf)
st.image(img, caption='聚类可视化', use_column_width=True)

# 交互式功能：鼠标悬停高亮簇
hovered_cluster = st.selectbox('选择一个簇进行高亮显示', range(n_clusters))

# 重新绘制图像，高亮指定簇
fig, ax = plt.subplots(figsize=(10, 8))
for cluster in range(n_clusters):
    alpha = 0.2 if cluster != hovered_cluster else 1.0
    subset = df[df['cluster'] == cluster]
    sns.scatterplot(
        x='x', y='y',
        data=subset,
        color=palette[cluster],
        s=60, edgecolor='w', linewidth=0.5,
        ax=ax,
        alpha=alpha
    )

# 标注簇中心
for i, center in enumerate(centers_reduced):
    ax.scatter(center[0], center[1], s=300, marker='*',
               c='gold', edgecolor='k', linewidth=1)

ax.set_title(f"2D 聚类分布 (共{n_clusters}个簇)", fontsize=12)
ax.set_xlabel('Feature 1', fontsize=10)
ax.set_ylabel('Feature 2', fontsize=10)

# 调整图例位置
legend = ax.legend(title='簇编号', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
legend.get_title().set_fontsize(8)

# 添加评估指标文字说明
ax.text(1.05, 0.02, textstr, transform=ax.transAxes,
        verticalalignment='bottom', fontsize=8)

# 保存高亮后的图像
canvas = FigureCanvas(fig)
canvas.draw()
buf = canvas.buffer_rgba()
img = Image.frombuffer('RGBA', canvas.get_width_height(), buf)
st.image(img, caption='高亮指定簇后的聚类可视化', use_column_width=True)