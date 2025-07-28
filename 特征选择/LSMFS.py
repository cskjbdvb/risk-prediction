import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


def lsmfs_feature_selection(X, y,
                            supplement_threshold=0.7,
                            feature_threshold=0.02,
                            top_k=None,
                            output_path=None):
    """
    修改后的LSMFS实现（自动处理完整标签数据）
    """
    y = np.array(y)
    assert y.ndim == 2, "y必须是二维标签矩阵"
    assert set(np.unique(y)) <= {0, 1}, "y必须为二进制形式"

    # ===== 修改点1：检测是否需要进行标签补充 =====
    if np.all((y == 0) | (y == 1)):
        print("检测到标签数据完整，跳过标签补充阶段")
        y_supp = y.copy()
    else:
        print("正在进行标签补充...")
        y_supp = y.copy()
        for label_idx in range(y.shape[1]):
            known_mask = y[:, label_idx] != -1  # 假设-1表示缺失
            if known_mask.sum() == 0 or known_mask.all():
                continue  # 跳过全缺失或全已知的标签

            X_train, X_test = X[known_mask], X[~known_mask]
            y_train = y[known_mask, label_idx]

            if len(np.unique(y_train)) < 2:
                continue

            knn = KNeighborsClassifier(n_neighbors=5)
            knn.fit(X_train, y_train)

            # ===== 修改点2：增加空数组检查 =====
            if len(X_test) > 0:  # 只有在有待补充样本时才预测
                proba = knn.predict_proba(X_test)[:, 1]
                supp_mask = (proba > supplement_threshold)
                y_supp[~known_mask, label_idx] = (proba > 0.5).astype(int)[supp_mask]

    # ===== 后续特征选择代码保持不变 =====
    mi_matrix = np.zeros((X.shape[1], y_supp.shape[1]))
    for i in range(y_supp.shape[1]):
        mi_matrix[:, i] = mutual_info_classif(X, y_supp[:, i], random_state=42)

    agg_scores = np.mean(mi_matrix, axis=1)

    if top_k is not None:
        selected_indices = np.argsort(agg_scores)[-top_k:]
    else:
        selected_indices = np.where(agg_scores >= feature_threshold)[0]

    selected_features = X.columns[selected_indices].tolist()

    result_df = pd.DataFrame({
        'Feature': X.columns,
        'LSMFS_Score': agg_scores,
        'Selected': [1 if x in selected_features else 0 for x in X.columns]
    }).sort_values('LSMFS_Score', ascending=False)

    if output_path:
        result_df.to_excel(output_path, index=False)

    return selected_features, result_df


# ==============================================
# 使用示例（根据您的数据调整以下部分）
# ==============================================
if __name__ == "__main__":
    # 可调整参数区域
    CONFIG = {
        'data_path': r"D:\研二2\论文撰写\3.30正式数据\KNN填充缺失值2（进行了异常值处理）.xlsx",
        'output_path': r"D:\研二2\论文撰写\3.30正式数据\LSMFS结果.xlsx",  # 自定义输出路径
        'supplement_threshold': 0.7,  # 标签补充置信度阈值(0-1)
        'feature_threshold': 0.02,  # 特征选择阈值
        'top_k': None,  # 指定需要选择的特征数量
    }

    # 1. 加载数据
    data = pd.read_excel(CONFIG['data_path'])
    X = data.iloc[:, 1:-4]  # 特征列（跳过ID和最后4列）
    y = data.iloc[:, -4:]  # 标签列（已经是二进制形式）

    # 2. 分类特征编码（如果存在）
    cat_cols = X.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    # 3. 执行LSMFS
    selected_features, result_df = lsmfs_feature_selection(
        X, y,
        supplement_threshold=CONFIG['supplement_threshold'],
        feature_threshold=CONFIG['feature_threshold'],
        top_k=CONFIG['top_k'],
        output_path=CONFIG['output_path']
    )

    # 4. 打印结果
    print(f"\n选中特征数: {len(selected_features)}")
    print("Top 10特征评分:")
    print(result_df.head(10))