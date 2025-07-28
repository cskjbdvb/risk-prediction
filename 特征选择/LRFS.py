import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import pairwise_distances


def lrfs_feature_selection(X, y, threshold=0.5, top_k=None, output_path=None):
    """
    LRFS (Label Redundancy-based Feature Selection)

    参数说明（可调整部分）:
    ----------------------------
    X: 特征DataFrame (n_samples, n_features)
    y: 多标签二进制矩阵 (n_samples, n_labels)
    threshold: 标签冗余阈值 (默认0.5)
        - 范围[0,1]，值越大认为标签冗余度越高
    top_k: 指定要选择的前K个特征 (默认None表示自动根据阈值选择)
    output_path: 结果保存路径 (默认None不保存)
        - 示例: r"D:\results\lrfs_features.xlsx"
    """
    # ========== 1. 数据校验 ==========
    y = np.array(y)
    assert y.ndim == 2, "y必须是二维标签矩阵"
    assert set(np.unique(y)) <= {0, 1}, "y必须为二进制形式"

    # ========== 2. 计算标签冗余矩阵 ==========
    # 使用Jaccard相似度衡量标签冗余
    label_sim = 1 - pairwise_distances(y.T, metric='hamming')  # 等价于Jaccard
    np.fill_diagonal(label_sim, 0)  # 对角线置零

    # ========== 3. 特征-标签互信息矩阵 ==========
    mi_matrix = np.zeros((X.shape[1], y.shape[1]))
    for i in range(y.shape[1]):
        mi_matrix[:, i] = mutual_info_classif(X, y[:, i], random_state=42)

    # ========== 4. 基于冗余度的特征评分 ==========
    feature_scores = np.zeros(X.shape[1])
    for j in range(X.shape[1]):  # 遍历每个特征
        # 计算特征对冗余标签的区分能力
        score = 0
        for k in range(y.shape[1]):
            for l in range(k + 1, y.shape[1]):
                if label_sim[k, l] > threshold:  # 只考虑冗余标签对
                    score += abs(mi_matrix[j, k] - mi_matrix[j, l])
        feature_scores[j] = score

    # ========== 5. 特征选择 ==========
    if top_k is not None:
        selected_indices = np.argsort(feature_scores)[-top_k:]  # 选择Top K
    else:
        selected_indices = np.where(feature_scores >= np.quantile(feature_scores, 0.75))[0]  # 默认选择前25%

    selected_features = X.columns[selected_indices].tolist()

    # ========== 6. 结果输出 ==========
    result_df = pd.DataFrame({
        'Feature': X.columns,
        'LRFS_Score': feature_scores,
        'Selected': [1 if x in selected_features else 0 for x in X.columns]
    }).sort_values('LRFS_Score', ascending=False)

    if output_path:
        result_df.to_excel(output_path, index=False)
        print(f"结果已保存到: {output_path}")

    return selected_features, result_df


# ==============================================
# 使用示例（根据您的数据调整以下部分）
# ==============================================
if __name__ == "__main__":
    # 可调整参数区域
    CONFIG = {
        'data_path': r"D:\\研二2\\论文撰写\\数据合并\\标准化、缺失值\\不填充缺失值\\独热编码.xlsx",
        'output_path': r"D:\\研二2\\论文撰写\\数据合并\\标准化、缺失值\\不填充缺失值\\特征选择\\LRFS.xlsx",  # 自定义输出路径
        'threshold': 0.5,  # 调整标签冗余阈值（0-1之间）
        'top_k': None,  # 指定需要选择的特征数量（如20）
    }

    # 1. 加载数据
    data = pd.read_excel(CONFIG['data_path'])
    X = data.iloc[:, 1:-4]  # 特征列（跳过ID和最后4列）
    y = data.iloc[:, -4:]  # 标签列（已经是二进制形式）

    # 2. 分类特征编码（如果存在）
    cat_cols = X.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    # 3. 执行LRFS
    selected_features, result_df = lrfs_feature_selection(
        X, y,
        threshold=CONFIG['threshold'],
        top_k=CONFIG['top_k'],
        output_path=CONFIG['output_path']
    )

    # 4. 打印结果
    print(f"\n选中特征数: {len(selected_features)}")
    print("Top 10特征评分:")
    print(result_df.head(10))