# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

######################################################
#                  用户配置区域                      #
######################################################
INPUT_PATH = "D:\\研二2\\论文撰写\\数据合并\\初处理完毕数据.xlsx"  # 预处理后的数据路径
OUTPUT_RESULT = "D:\\研二2\\论文撰写\\数据合并\\互信息结果.xlsx"  # 特征重要性结果路径
N_NEIGHBORS = 5  # 近邻数（建议5-20）
SAMPLE_SIZE = None  # 抽样数量（大数据时启用，如10000）
RANDOM_STATE = 42  # 随机种子


######################################################

def multilabel_mi_importance(X, y, n_neighbors=5):
    """
    多标签互信息特征重要性计算（基于联合分布）

    参数:
    X : 特征矩阵 (n_samples, n_features)
    y : 多标签目标 (n_samples, n_labels)
    n_neighbors : 近邻数（用于密度估计）

    返回:
    mi_scores : 各特征的重要性得分
    """
    # 将多标签编码为联合分布（笛卡尔积编码）
    y_combined = np.apply_along_axis(
        lambda x: '_'.join(x.astype(str)),
        axis=1,
        arr=y.astype(int)
    )
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_combined)

    # 计算互信息（基于联合标签编码）
    mi_scores = mutual_info_classif(
        X, y_encoded,
        n_neighbors=n_neighbors,
        random_state=RANDOM_STATE
    )

    return mi_scores


def main():
    # 1. 加载数据
    df = pd.read_excel(INPUT_PATH)
    print(f"数据维度: {df.shape}")

    # 2. 数据抽样（可选）
    if SAMPLE_SIZE and len(df) > SAMPLE_SIZE:
        df = df.sample(SAMPLE_SIZE, random_state=RANDOM_STATE)
        print(f"已随机抽样{SAMPLE_SIZE}条数据")

    # 3. 分离数据
    id_col = df.columns[0]  # 第一列是ID
    label_cols = df.columns[-4:]  # 最后四列是标签
    feature_cols = df.columns[1:-4]  # 中间列为特征

    X = df[feature_cols].values.astype(np.float32)
    y = df[label_cols].values.astype(int)

    # 4. 计算多标签互信息
    print("\n正在计算多标签联合互信息...")
    mi_scores = multilabel_mi_importance(X, y, N_NEIGHBORS)

    # 5. 构建结果DataFrame
    result_df = pd.DataFrame({
        "Feature": feature_cols,
        "MultiLabel_MI": mi_scores
    }).sort_values("MultiLabel_MI", ascending=False)

    # 6. 保存结果
    result_df.to_excel(OUTPUT_RESULT, index=False)
    print(f"\n特征重要性结果已保存至: {OUTPUT_RESULT}")
    print("Top 50重要特征:")
    print(result_df.head(50))


if __name__ == "__main__":
    main()