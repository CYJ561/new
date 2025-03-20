import numpy as np
from scipy.spatial import KDTree
from matplotlib.patches import Circle
from tqdm import tqdm
import matplotlib.pyplot as plt


class OverlapEvaluator:
    def __init__(self, positions, thumbnail_size=50):
        """
        改进的重叠评估器
        参数：
            positions: 图像中心坐标数组，形状为(N, 2)
            thumbnail_size: 缩略图边长（像素）
        """
        self.positions = np.array(positions)
        num_points = len(positions)
        # 根据数据点数量动态调整缩略图边长
        min_size = 0.001
        max_size = 200
        # 简单的调整策略，数据点越多，边长越小
        self.R = max(min_size, min(max_size, thumbnail_size / np.sqrt(num_points))) / 2
        self.kd_tree = KDTree(self.positions)

    def _calculate_overlap(self, d):
        """计算两个圆的重叠面积（优化版）"""
        if d >= 2 * self.R:
            return 0.0

        ratio = d / (2 * self.R)
        theta = 2 * np.arccos(ratio)
        return self.R ** 2 * (theta - np.sin(theta))

    def compute_visibility_cost(self, show_progress=True):
        """计算总可见性成本（并行优化）"""
        n = len(self.positions)
        total_pairs = n * (n - 1) // 2
        sum_visibility = 0.0

        # 使用KDTree快速查询邻近对
        pairs = self.kd_tree.query_pairs(2 * self.R)

        # 进度条显示
        if show_progress:
            print(f"正在计算 {len(pairs)} 个邻近对...")
            iter_obj = tqdm(pairs, desc="处理图像对")
        else:
            iter_obj = pairs

        # 计算实际重叠对
        for i, j in iter_obj:
            d = np.linalg.norm(self.positions[i] - self.positions[j])
            overlap = self._calculate_overlap(d)
            sum_visibility += (1 - overlap / (np.pi * self.R ** 2))

        # 计算未重叠对贡献 (1 - 0/area = 1)
        non_overlap_pairs = total_pairs - len(pairs)
        sum_visibility += non_overlap_pairs * 1.0

        return sum_visibility / total_pairs

    def visualize(self, highlight_pairs=True):
        """生成可视化报告"""
        num_points = len(self.positions)
        # 根据数据点数量进一步动态调整点的大小，让直径变小
        point_size = max(0.1, 500 / num_points)
        alpha = min(0.8, 100 / num_points)

        plt.figure(figsize=(12, 8))
        ax = plt.gca()

        # 绘制所有图像位置
        plt.scatter(self.positions[:, 0], self.positions[:, 1],
                    s=point_size, alpha=alpha, edgecolor='k')

        # 绘制缩略图边界圆，进一步缩小圈的大小
        circle_linewidth = max(0.1, 1 / np.sqrt(num_points))
        for x, y in self.positions:
            ax.add_patch(Circle((x, y), self.R,
                                fill=False, color='blue', alpha=0.3, linewidth=circle_linewidth))

        # 高亮重叠区域，让相连的线变细
        line_width = max(0.1, 0.5/ np.sqrt(num_points))
        if highlight_pairs:
            pairs = self.kd_tree.query_pairs(2 * self.R)
            for i, j in pairs:
                plt.plot([self.positions[i][0], self.positions[j][0]],
                         [self.positions[i][1], self.positions[j][1]],
                         color='red', alpha=0.4, linewidth=line_width)

        plt.title(f"图像布局可视化 (R={self.R * 2:.6f}px)")
        plt.xlabel("X 坐标")
        plt.ylabel("Y 坐标")
        plt.grid(True)
        return plt