import streamlit as st
import numpy as np
import os
from PIL import Image
from LabColorHistogram import LabColorHistogram
from ResNetFeatureExtractor import ResNetFeatureExtractor
from K_means import kmeans_classify
import matplotlib.pyplot as plt
import umap
from overlap_Cv import OverlapEvaluator
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import Hubert
import pandas as pd
import plotly.express as px

# Streamlit页面配置
st.set_page_config(page_title="图像可视化分析", layout="wide")


def main():
    st.title("📸 图像特征可视化分析系统")

    # ========== 侧边栏控制 ==========
    st.sidebar.header("⚙️ 参数配置")
    data_path = st.sidebar.text_input("图像目录路径",
                                      "D:/Image_visualization3/Image_visualization/Image_visualization/Source/")
    feature_method = st.sidebar.selectbox("特征提取方法",
                                          ("LAB颜色直方图", "ResNet18特征"))
    n_neighbors = st.sidebar.slider("UMAP 近邻数", 5, 50, 15)  # UMAP 的近邻数参数
    min_dist = st.sidebar.slider("UMAP 最小距离", 0.0, 1.0, 0.1)  # UMAP 的最小距离参数
    k_clusters = st.sidebar.slider("聚类数量", 2, 15, 8)
    thumbnail_size = st.sidebar.slider("缩略图边长", 20, 2000, 50)

    # 初始化表格显示
    if os.path.exists('analysis_results.csv'):
        results = pd.read_csv('analysis_results.csv')
    else:
        results = pd.DataFrame(columns=["数据路径", "特征提取方法", "UMAP近邻数", "UMAP最小距离", "聚类数量", "缩略图边长",
                                        "轮廓系数", "Calinski-Harabasz指数", "Davies-Bouldin指数", "Hubert指数", "总可见性成本Cv"])
    st.subheader("分析结果记录")
    table = st.table(results)

    # ========== 主处理流程 ==========
    if st.button("🚀 开始分析", type="primary"):
        with st.spinner('正在处理中...'):
            try:
                # 特征提取
                if feature_method == "LAB颜色直方图":
                    extractor = LabColorHistogram(data_path)
                    features = extractor.Lab()
                    valid_count = features.shape[0]
                else:
                    extractor = ResNetFeatureExtractor()
                    features, valid_images = extractor.batch_extract(data_path)
                    valid_count = len(valid_images)

                # 有效性检查
                if valid_count < 2:
                    raise ValueError("至少需要2个有效数据点进行分析")

                st.success(f"✅ 成功加载 {valid_count} 个有效数据点")

                # 降维处理
                with st.status("正在进行降维处理..."):
                    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=2)
                    umap_result = reducer.fit_transform(features)

                # 聚类分析
                with st.status("正在进行聚类分析..."):
                    centers, _, cluster_idx = kmeans_classify(umap_result, k_clusters)

                # ========== 可视化结果 ==========
                st.subheader("📊 分析结果")

                # 创建两列布局
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("### 聚类可视化")
                    labels = np.zeros(valid_count, dtype=int)
                    for i, indices in enumerate(cluster_idx):
                        labels[indices] = i

                    # 创建DataFrame用于plotly
                    df = pd.DataFrame({
                        'UMAP 维度1': umap_result[:, 0],
                        'UMAP 维度2': umap_result[:, 1],
                        '簇编号': labels
                    })

                    # 创建交互式散点图
                    fig = px.scatter(df, x='UMAP 维度1', y='UMAP 维度2', color='簇编号',
                                     hover_data=['簇编号'],
                                     opacity=0.5,
                                     color_continuous_scale=px.colors.sequential.Viridis)

                    # 定义鼠标悬停事件的回调函数
                    def update_trace(trace, points, selector):
                        if points.point_inds:
                            hovered_cluster = df.iloc[points.point_inds[0]]['簇编号']
                            new_opacities = [1 if cluster == hovered_cluster else 0.1 for cluster in df['簇编号']]
                            fig.update_traces(opacity=new_opacities)
                        else:
                            fig.update_traces(opacity=0.5)

                    # 绑定鼠标悬停事件
                    fig.data[0].on_hover(update_trace)

                    st.plotly_chart(fig)

                with col2:
                    st.markdown("### 图像分布图")
                    bk_img = visualize_images(umap_result, data_path)
                    st.image(bk_img, caption="图像空间分布", use_column_width=True)

                # 计算重叠指标
                evaluator = OverlapEvaluator(umap_result, thumbnail_size=thumbnail_size)
                cv = evaluator.compute_visibility_cost()
                st.info(f"总可见性成本 Cv = {cv:.4f}")

                # 可视化重叠指标
                st.subheader("📈 图像重叠可视化")
                fig = evaluator.visualize()
                st.pyplot(fig)

                # 计算聚类评估指标
                labels = np.zeros(valid_count, dtype=int)
                for i, indices in enumerate(cluster_idx):
                    labels[indices] = i

                silhouette = silhouette_score(umap_result, labels)
                calinski_harabasz = calinski_harabasz_score(umap_result, labels)
                davies_bouldin = davies_bouldin_score(umap_result, labels)
                hubert = Hubert.hubert_cluster(umap_result, centers, cluster_idx, k_clusters)

                # 显示聚类评估指标
                st.subheader("📋 聚类评估指标")
                st.write(f"轮廓系数  [-1,1]值越大越好: {silhouette:.5f}")
                st.write(f"CH指数 值越大分离度越好: {calinski_harabasz:.5f}")
                st.write(f"DB指数 值越小聚类越紧凑: {davies_bouldin:.5f}")
                st.write(f"Hubert指数: {hubert:.9f}")

                # 绘制轮廓图
                st.subheader("📊 聚类轮廓图")
                from sklearn.metrics import silhouette_samples
                sample_silhouette_values = silhouette_samples(umap_result, labels)
                y_lower = 10
                plt.figure(figsize=(10, 8))
                for i in range(k_clusters):
                    # Aggregate the silhouette scores for samples belonging to
                    # cluster i, and sort them
                    ith_cluster_silhouette_values = \
                        sample_silhouette_values[labels == i]

                    ith_cluster_silhouette_values.sort()

                    size_cluster_i = ith_cluster_silhouette_values.shape[0]
                    y_upper = y_lower + size_cluster_i

                    color = plt.cm.nipy_spectral(float(i) / k_clusters)
                    plt.fill_betweenx(np.arange(y_lower, y_upper),
                                      0, ith_cluster_silhouette_values,
                                      facecolor=color, edgecolor=color, alpha=0.7)

                    # Label the silhouette plots with their cluster numbers at the middle
                    plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                    # Compute the new y_lower for next plot
                    y_lower = y_upper + 10  # 10 for the 0 samples

                plt.title("轮廓系数分布图.")
                plt.xlabel("轮廓系数值")
                plt.ylabel("簇样本排序")

                # The vertical line for average silhouette score of all the values
                plt.axvline(x=silhouette, color="red", linestyle="--")

                plt.yticks([])  # Clear the yaxis labels / ticks
                plt.xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

                st.pyplot(plt)

                # 记录结果
                result = {
                    "数据路径": data_path,
                    "特征提取方法": feature_method,
                    "UMAP近邻数": n_neighbors,
                    "UMAP最小距离": min_dist,
                    "聚类数量": k_clusters,
                    "缩略图边长": thumbnail_size,
                    "轮廓系数": silhouette,
                    "Calinski-Harabasz指数": calinski_harabasz,
                    "Davies-Bouldin指数": davies_bouldin,
                    "Hubert指数": hubert,
                    "总可见性成本Cv": cv
                }

                result_df = pd.DataFrame([result])
                if not os.path.exists('analysis_results.csv'):
                    result_df.to_csv('analysis_results.csv', index=False)
                else:
                    result_df.to_csv('analysis_results.csv', mode='a', header=False, index=False)

                # 更新表格
                updated_results = pd.read_csv('analysis_results.csv')
                table.data = updated_results

            except Exception as e:
                st.error(f"❌ 分析失败: {str(e)}")
                st.info("💡 可能的原因：1. 图像路径错误 2. 图像格式不支持 3. 有效数据点不足")


def visualize_images(coords, img_path):
    """生成图像分布图"""
    bk_width, bk_height = 4000, 3000
    max_dim = 200
    back = Image.new('RGB', (bk_width, bk_height), (255, 255, 255))

    # 归一化坐标
    tx = (coords[:, 0] - np.min(coords[:, 0])) / (np.max(coords[:, 0]) - np.min(coords[:, 0]))
    ty = (coords[:, 1] - np.min(coords[:, 1])) / (np.max(coords[:, 1]) - np.min(coords[:, 1]))

    images = os.listdir(img_path)
    for img_name, x, y in zip(images, tx, ty):
        try:
            img_path_full = os.path.join(img_path, img_name)
            if not img_path_full.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            img = Image.open(img_path_full)
            rs = max(1, img.width / max_dim, img.height / max_dim)
            img = img.resize((int(img.width / rs), int(img.height / rs)))
            back.paste(img, (int((bk_width - max_dim) * x), int((bk_height - max_dim) * y)))
        except Exception as e:
            st.warning(f"⚠️ 无法加载图像 {img_name}: {str(e)}")
    return back


if __name__ == "__main__":
    main()