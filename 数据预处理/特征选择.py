import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# 1. 加载数据
def load_data(file_path):
    """
    加载Excel数据
    :param file_path: Excel文件路径
    :return: 特征矩阵X和标签矩阵y，以及特征列名
    """
    # 读取Excel文件
    df = pd.read_excel(file_path)

    # 假设第一列是ID号，最后四列是标签
    # 特征列是除了第一列和最后四列之外的所有列
    feature_columns = df.columns[1:-4]  # 第一列是ID号，最后四列是标签
    label_columns = df.columns[-4:]  # 最后四列是标签

    # 分离特征和标签
    X = df[feature_columns].values
    y = df[label_columns].values

    return X, y, feature_columns


# 2. ML-ReliefF算法
def ml_relief_f(X, y, k=10, m=100):
    """
    ML-ReliefF算法实现
    :param X: 特征矩阵 (n_samples, n_features)
    :param y: 标签矩阵 (n_samples, n_labels)
    :param k: 近邻数量
    :param m: 随机选择的样本数量
    :return: 特征权重 (n_features,)
    """
    n_samples, n_features = X.shape
    n_labels = y.shape[1]

    # 初始化特征权重
    weights = np.zeros(n_features)

    for _ in range(m):
        # 随机选择一个样本
        idx = np.random.randint(n_samples)
        R = X[idx]
        R_labels = y[idx]

        # 找到与R同类的样本
        same_class_indices = np.where(np.all(y == R_labels, axis=1))[0]
        if len(same_class_indices) > 1:
            same_class_indices = same_class_indices[same_class_indices != idx]
        if len(same_class_indices) > k:
            same_class_indices = np.random.choice(same_class_indices, k, replace=False)

        # 找到与R不同类的样本
        diff_class_indices = np.where(np.any(y != R_labels, axis=1))[0]
        if len(diff_class_indices) > k:
            diff_class_indices = np.random.choice(diff_class_indices, k, replace=False)

        # 计算与R的最近邻
        neighbors = np.concatenate((same_class_indices, diff_class_indices))
        distances = np.linalg.norm(X[neighbors] - R, axis=1)
        nearest_neighbors = neighbors[np.argsort(distances)][:k]

        # 更新特征权重
        for feature_idx in range(n_features):
            for neighbor_idx in nearest_neighbors:
                neighbor_labels = y[neighbor_idx]
                diff = abs(R[feature_idx] - X[neighbor_idx, feature_idx])
                if np.all(neighbor_labels == R_labels):
                    weights[feature_idx] -= diff / (k * m)
                else:
                    shared_labels = np.sum(np.logical_and(R_labels, neighbor_labels))
                    weights[feature_idx] += (shared_labels / (n_labels - shared_labels)) * diff / (k * m)

    return weights


# 3. 特征选择
def select_features_by_ml_relief_f(X, y, feature_columns, k=10, m=100):
    """
    使用ML-ReliefF方法选择特征
    :param X: 特征矩阵 (n_samples, n_features)
    :param y: 标签矩阵 (n_samples, n_labels)
    :param feature_columns: 特征列名
    :param k: 近邻数量
    :param m: 随机选择的样本数量
    :return: 选择的特征索引和对应的权重
    """
    # 计算特征权重
    weights = ml_relief_f(X, y, k=k, m=m)

    # 根据权重对特征进行排序
    sorted_indices = np.argsort(weights)[::-1]  # 降序排序
    sorted_weights = weights[sorted_indices]  # 对应的权重

    # 输出每个特征的权重
    print("所有特征的权重（从大到小）：")
    for idx, weight in zip(sorted_indices, sorted_weights):
        print(f"特征 {feature_columns[idx]} 的权重: {weight:.4f}")

    return sorted_indices, sorted_weights


# 4. 评估不同k和m值的效果
def evaluate_k_m_values(X, y, feature_columns, k_values, m_values):
    """
    评估不同k和m值的效果
    :param X: 特征矩阵 (n_samples, n_features)
    :param y: 标签矩阵 (n_samples, n_labels)
    :param feature_columns: 特征列名
    :param k_values: k值列表
    :param m_values: m值列表
    :return: 最佳k和m值
    """
    # 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 初始化最佳参数和准确率
    best_k = None
    best_m = None
    best_accuracy = 0
    best_selected_indices = None
    best_weights = None

    for k in k_values:
        for m in m_values:
            # 使用ML-ReliefF算法选择特征
            selected_indices, weights = select_features_by_ml_relief_f(X_train, y_train, feature_columns, k=k, m=m)

            # 使用选择的特征训练分类器
            X_train_selected = X_train[:, selected_indices]
            X_test_selected = X_test[:, selected_indices]

            classifier = RandomForestClassifier(random_state=42)
            classifier.fit(X_train_selected, y_train)
            y_pred = classifier.predict(X_test_selected)

            # 计算准确率
            accuracy = accuracy_score(y_test, y_pred)

            # 更新最佳k和m值
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_k = k
                best_m = m
                best_selected_indices = selected_indices
                best_weights = weights

    print(f"最佳k值: {best_k}")
    print(f"最佳m值: {best_m}")
    print(f"最佳准确率: {best_accuracy}")
    return best_k, best_m, best_selected_indices, best_weights


# 5. 主函数
if __name__ == "__main__":
    # 加载数据
    file_path = 'D:\\研二2\\论文撰写\\标准化、填充缺失值后的数据.xlsx'  # 替换为你的Excel文件路径
    X, y, feature_columns = load_data(file_path)

    # 定义不同的k和m值
    k_values = [5, 10, 20, 30]
    m_values = [50, 100, 200,300]

    # 评估不同k和m值的效果
    best_k, best_m, best_selected_indices, best_weights = evaluate_k_m_values(X, y, feature_columns, k_values, m_values)

    # 输出最佳k和m值下的特征选择结果
    print("\n最佳k和m值下的特征选择结果：")
    print("选择的特征索引:", best_selected_indices)
    print("选择的特征名称:", feature_columns[best_selected_indices])
    print("选择的特征权重:", best_weights)