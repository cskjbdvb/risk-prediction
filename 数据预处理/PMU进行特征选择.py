import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif


# 1. 加载数据
def load_data(file_path):
    """
    加载Excel数据
    :param file_path: Excel文件路径
    :return: 特征矩阵X和标签矩阵y
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


# 2. 计算互信息
def calculate_mutual_information(X, y):
    """
    计算特征与标签集合之间的互信息
    :param X: 特征矩阵 (n_samples, n_features)
    :param y: 标签矩阵 (n_samples, n_labels)
    :return: 每个特征与标签集合之间的互信息值
    """
    # 将多标签数据转换为单标签数据，用于互信息计算
    # 通过将每个标签视为一个二进制位，将多标签组合成一个整数
    y_combined = np.packbits(y.astype(int), axis=1).flatten()

    # 计算互信息
    mi_scores = mutual_info_classif(X, y_combined, discrete_features=False)

    return mi_scores


# 3. 特征选择
def select_features_by_pmu(X, y, feature_columns, k):
    """
    使用PMU方法选择特征
    :param X: 特征矩阵 (n_samples, n_features)
    :param y: 标签矩阵 (n_samples, n_labels)
    :param feature_columns: 特征列名
    :param k: 选择的特征数量
    :return: 选择的特征索引和对应的互信息值
    """
    # 计算互信息
    mi_scores = calculate_mutual_information(X, y)

    # 根据互信息值对特征进行排序
    sorted_indices = np.argsort(mi_scores)[::-1]  # 降序排序
    sorted_mi_scores = mi_scores[sorted_indices]  # 对应的互信息值

    # 选择前k个特征
    selected_indices = sorted_indices[:k]
    selected_mi_scores = sorted_mi_scores[:k]

    # 输出每个特征的互信息值
    print("所有特征的互信息值（从大到小）：")
    for idx, mi_score in zip(sorted_indices, sorted_mi_scores):
        print(f"特征 {feature_columns[idx]} 的互信息值: {mi_score:.4f}")

    return selected_indices, selected_mi_scores


# 4. 主函数
if __name__ == "__main__":
    # 加载数据
    file_path = 'D:\\研二2\\论文撰写\\标准化、填充缺失值后的数据.xlsx'  # 替换为你的Excel文件路径
    X, y, feature_columns = load_data(file_path)

    # 特征选择
    k = 100  # 选择的特征数量
    selected_indices, selected_mi_scores = select_features_by_pmu(X, y, feature_columns, k)

    # 输出结果
    print("\n选择的特征索引:", selected_indices)
    print("选择的特征名称:", feature_columns[selected_indices])
    print("选择的特征互信息值:", selected_mi_scores)
