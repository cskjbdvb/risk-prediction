import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder


def information_gain_multi_label(X, y, threshold=0.01):
    """
    IGMF (Information Gain for Multi-label Feature Selection)
   

    参数:
        X: 特征DataFrame (n_samples, n_features)
        y: 已经是二进制形式的多标签数据 (n_samples, n_labels)
        threshold: 信息增益阈值 (默认0.01)

    返回:
        selected_features: 选中的特征列表
        ig_scores: 各特征的信息增益得分
    """
  
    y = np.array(y)

    # 2. 检查y的二进制格式
    assert set(np.unique(y)) <= {0, 1}, "y必须已经是二进制形式（只包含0和1）"
    assert y.ndim == 2, "y必须是二维矩阵"

    # 3. 计算每个特征对每个标签的信息增益
    ig_matrix = np.zeros((X.shape[1], y.shape[1]))
    for i in range(y.shape[1]):
        ig_matrix[:, i] = mutual_info_classif(X, y[:, i], random_state=42)

    # 4. 聚合信息增益（平均）
    ig_scores = np.mean(ig_matrix, axis=1)

    # 5. 根据阈值选择特征
    selected_indices = np.where(ig_scores >= threshold)[0]
    selected_features = X.columns[selected_indices].tolist()

    return selected_features, ig_scores


# 使用示例
if __name__ == "__main__":
    # 加载数据
    data = pd.read_excel("独热编码.xlsx")

    # 分离特征和标签
    X = data.iloc[:, 1:-4]  # 特征列（跳过ID和最后4列标签）
    y = data.iloc[:, -4:]  # 已经是二进制形式的4个标签列

    # 分类特征标签编码（如果存在分类特征）
    cat_cols = X.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    # 执行IGMF特征选择
    threshold = 0.01  # 可调整的阈值
    selected_features, ig_scores = information_gain_multi_label(X, y, threshold)

    # 输出结果
    print(f"原始特征数: {X.shape[1]}")
    print(f"选中特征数: {len(selected_features)}")
    print("\n信息增益得分:")
    for feat, score in zip(X.columns, ig_scores):
        print(f"{feat}: {score:.4f}")

    print("\nSelected Features:")
    print(selected_features)

    # 保存选中的特征和对应的得分
    result_df = pd.DataFrame({
        'Feature': X.columns,
        'IG_Score': ig_scores
    }).sort_values('IG_Score', ascending=False)

    result_df.to_excel("feature_ranking.xlsx", index=False)
    print("\n结果已保存到 feature_ranking.xlsx")
