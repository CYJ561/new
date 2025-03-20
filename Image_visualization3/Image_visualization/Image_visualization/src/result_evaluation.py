import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from sklearn.datasets import make_blobs

plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

class ClusteringVisualReport:
    def __init__(self, data, labels, feature_names=None):
        """
        可视化报告生成器
        参数：
            data: 二维数据集 (n_samples, n_features)
            labels: 聚类标签数组 (n_samples,)
            feature_names: 特征名称列表
        """
        self.data = data
        self.labels = labels
        self.n_clusters = len(np.unique(labels))
        self.feature_names = feature_names or ['Feature 1', 'Feature 2']
        # 颜色配置
        self.palette = sns.color_palette("husl", self.n_clusters)

    def generate_report(self, figsize=(25, 18)):
        """生成完整可视化报告"""
        fig = plt.figure(figsize=figsize, dpi=100)
        gs = GridSpec(3, 3, figure=fig, wspace=0.4, hspace=0.4)

        # 绘制各子图
        cluster_ax = fig.add_subplot(gs[:2, :2])
        silhouette_ax = fig.add_subplot(gs[0, 2])
        metric_ax = fig.add_subplot(gs[1, 2], polar=True)
        distance_ax = fig.add_subplot(gs[2, :])

        self._plot_cluster_distribution(cluster_ax)
        self._plot_silhouette_distribution(silhouette_ax)
        self._plot_metric_comparison(metric_ax)
        self._plot_intra_inter_distances(distance_ax)

        plt.tight_layout()
        return fig

    def _plot_cluster_distribution(self, ax):
        """绘制聚类分布图"""
        scatter = sns.scatterplot(
            x=self.data[:, 0], y=self.data[:, 1],
            hue=self.labels, palette=self.palette,
            s=60, edgecolor='w', linewidth=0.5,
            ax=ax
        )

        # 标注簇中心
        centers = []
        for cluster_id in range(self.n_clusters):
            cluster_data = self.data[self.labels == cluster_id]
            center = np.mean(cluster_data, axis=0)
            centers.append(center)
            ax.scatter(center[0], center[1], s=300, marker='*',
                       c='gold', edgecolor='k', linewidth=1)

        ax.set_title(f"2D 聚类分布 (共{self.n_clusters}个簇)", fontsize=12)
        ax.set_xlabel(self.feature_names[0], fontsize=10)
        ax.set_ylabel(self.feature_names[1], fontsize=10)

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

    def _plot_silhouette_distribution(self, ax):
        """绘制轮廓系数分布"""
        from sklearn.metrics import silhouette_samples

        silhouette_vals = silhouette_samples(self.data, self.labels)
        y_lower = 10

        for i in range(self.n_clusters):
            ith_cluster_silhouette_vals = silhouette_vals[self.labels == i]
            ith_cluster_silhouette_vals.sort()

            size_cluster_i = ith_cluster_silhouette_vals.shape[0]
            y_upper = y_lower + size_cluster_i

            color = self.palette[i]
            ax.fill_betweenx(
                np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_vals,
                facecolor=color, edgecolor=color, alpha=0.7
            )

            # 标注簇编号
            ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i), fontsize=8)
            y_lower = y_upper + 10

        ax.set_title("轮廓系数分布图", fontsize=10)
        ax.set_xlabel("轮廓系数值", fontsize=10)
        ax.set_ylabel("簇样本排序", fontsize=10)
        ax.axvline(x=np.mean(silhouette_vals), color='red', linestyle='--')
        ax.text(0.6, 0.95, f"平均值: {np.mean(silhouette_vals):.2f}",
                transform=ax.transAxes, color='red', fontsize=8)

    def _plot_metric_comparison(self, ax):
        """绘制指标对比雷达图"""
        from sklearn.metrics import (
            silhouette_score, calinski_harabasz_score, davies_bouldin_score
        )

        # 计算指标值
        metrics = {
            '轮廓系数': silhouette_score(self.data, self.labels),
            'CH指数': calinski_harabasz_score(self.data, self.labels),
            'DB指数': davies_bouldin_score(self.data, self.labels)
        }

        # 归一化处理
        max_values = {
            '轮廓系数': 1.0,
            'CH指数': 1000,  # 根据实际情况调整
            'DB指数': 2.0
        }

        normalized = {
            k: v / max_values[k] for k, v in metrics.items()
        }

        # 雷达图绘制
        labels = list(normalized.keys())
        values = list(normalized.values())
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        values += values[:1]
        angles += angles[:1]

        ax.plot(angles, values, color='b', linewidth=1)
        ax.fill(angles, values, color='b', alpha=0.25)
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=8)
        ax.set_title("指标归一化对比", y=1.1, fontsize=10)

    def _plot_intra_inter_distances(self, ax):
        """绘制簇内/簇间距离分布"""
        intra_dists = []
        inter_dists = []

        # 计算各簇内距离
        for cluster_id in range(self.n_clusters):
            cluster_data = self.data[self.labels == cluster_id]
            centroid = np.mean(cluster_data, axis=0)
            dists = np.linalg.norm(cluster_data - centroid, axis=1)
            intra_dists.extend(dists)

        # 计算簇间距离
        centroids = []
        for cluster_id in range(self.n_clusters):
            centroids.append(np.mean(self.data[self.labels == cluster_id], axis=0))

        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                inter_dists.append(np.linalg.norm(centroids[i] - centroids[j]))

        # 绘制箱线图
        sns.boxplot(
            data=[intra_dists, inter_dists],
            palette=['skyblue', 'lightgreen'],
            width=0.4,
            ax=ax
        )
        ax.set_xticklabels(['簇内距离', '簇间距离'], fontsize=10)
        ax.set_title("距离分布对比", fontsize=12)
        ax.set_ylabel("欧氏距离", fontsize=10)

# 示例用法 --------------------------------------------------
if __name__ == "__main__":
    # 生成示例数据
    X, y = make_blobs(
        n_samples=500,
        n_features=2,
        centers=4,
        cluster_std=0.8,
        random_state=42
    )

    # 创建报告生成器
    report = ClusteringVisualReport(X, y, ['特征1', '特征2'])

    # 生成可视化报告
    fig = report.generate_report()

    # 保存报告
    fig.savefig('clustering_report.png', bbox_inches='tight', pad_inches=0.3)
    plt.show()