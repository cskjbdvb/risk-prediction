# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif

######################################################
#                  用户配置区域                      #
######################################################
INPUT_PATH = "D:\\研二2\\论文撰写\\17-23年数据\\标准化.xlsx"  # 输入标准化训练集路径
OUTPUT_MI_PATH = "D:\\研二2\\论文撰写\\17-23年数据\\互信息.xlsx"  # 互信息结果保存路径
OUTPUT_TOP_FEATURES = "D:\\研二2\\论文撰写\\17-23年数据\\互信息几个重要的.xlsx"  # 筛选后的重要特征保存路径
TOP_K = 20  # 可视化显示每个标签前20个特征
MI_THRESHOLD = 0.01  # 特征筛选阈值（MI值>0.01保留）


######################################################

def main():
    # 1. 加载数据
    df = pd.read_excel(INPUT_PATH)
    print(f"数据维度: {df.shape}")

    # 2. 分离特征和标签
    id_col = df.columns[0]  # 第一列为ID
    feature_cols = df.columns[1:-4].tolist()  # 中间列为特征
    label_cols = df.columns[-4:].tolist()  # 最后四列为标签

    X = df[feature_cols].values.astype(np.float32)
    y = df[label_cols].values.astype(int)

    # 3. 检查常量特征（方差为0）
    constant_mask = X.var(axis=0) == 0
    if np.any(constant_mask):
        print(f"发现 {sum(constant_mask)} 个常量特征，已自动移除:")
        print(np.array(feature_cols)[constant_mask])
        X = X[:, ~constant_mask]
        feature_cols = np.array(feature_cols)[~constant_mask].tolist()

    # 4. 计算每个标签的互信息
    mi_results = []
    for idx, label_name in enumerate(label_cols):
        print(f"\n正在计算标签 [{label_name}] 的互信息...")
        mi = mutual_info_classif(X, y[:, idx], random_state=42)
        mi_results.append(mi)

    # 5. 构建结果DataFrame
    result_df = pd.DataFrame(
        np.array(mi_results).T,
        columns=label_cols,
        index=feature_cols
    )
    result_df["Average_MI"] = result_df.mean(axis=1)
    result_df = result_df.sort_values("Average_MI", ascending=False)

    # 6. 保存结果
    result_df.to_excel(OUTPUT_MI_PATH)
    print(f"\n互信息结果已保存至: {OUTPUT_MI_PATH}")

    # 7. 筛选重要特征
    selected_features = result_df[result_df["Average_MI"] > MI_THRESHOLD].index.tolist()
    pd.Series(selected_features).to_csv(OUTPUT_TOP_FEATURES, index=False)
    print(f"筛选出 {len(selected_features)} 个重要特征（MI > {MI_THRESHOLD}），已保存至: {OUTPUT_TOP_FEATURES}")

    # 8. 可视化
    plt.figure(figsize=(15, 10))
    for i, label in enumerate(label_cols, 1):
        plt.subplot(2, 2, i)
        # 取当前标签的Top K特征
        top_features = result_df.sort_values(label, ascending=False).head(TOP_K).index
        sns.barplot(
            x=result_df.loc[top_features, label],
            y=top_features,
            palette="Blues_d"
        )
        plt.title(f"Top {TOP_K} Features for {label}")
        plt.xlabel("Mutual Information Score")
    plt.tight_layout()
    plt.savefig("mi_visualization.png")
    plt.show()


if __name__ == "__main__":
    main()