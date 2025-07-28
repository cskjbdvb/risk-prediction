# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

######################################################
#                  用户配置区域                      #
######################################################
INPUT_PATH = "D:\\研二2\\论文撰写\\数据合并\\2025.4.17特征选择之后重新选择数据\\原始数据.xlsx"  # 输入数据路径
OUTPUT_PATH = "D:\\研二2\\论文撰写\\数据合并\\2025.4.17特征选择之后重新选择数据\\箱线图异常值处理.xlsx"  # 输出路径
OUTLIER_METHOD = "remove"  # 处理方式: "remove"/"clip"/"median"
PLOT_BEFORE_AFTER = True  # 是否生成处理前后对比图

# ========= 新增手动指定功能 =========
AUTO_THRESHOLD = 10  # 自动检测阈值（唯一值≥此值视为连续型）
MANUAL_CONTINUOUS = []  # 强制视为连续型的特征
MANUAL_NON_CONTINUOUS = []  # 强制视为非连续型的特征


######################################

def detect_continuous_features(df):
    """智能识别连续型特征（结合自动检测和手动指定）"""
    auto_continuous = []
    for col in df.columns:
        if df[col].nunique() >= AUTO_THRESHOLD and np.issubdtype(df[col].dtype, np.number):
            auto_continuous.append(col)

    # 合并手动指定
    final_continuous = list(
        set(auto_continuous) |
        set(MANUAL_CONTINUOUS)
    )

    # 排除手动指定为非连续型的特征
    final_continuous = [
        col for col in final_continuous
        if col not in MANUAL_NON_CONTINUOUS
    ]

    # 有效性检查
    missing_manual = set(MANUAL_CONTINUOUS) - set(df.columns)
    if missing_manual:
        print(f"警告：手动指定的连续型特征不存在 {missing_manual}")

    return sorted(final_continuous)


def process_outliers(df, method="remove"):
    """处理异常值（返回处理后的DataFrame和处理统计信息）"""
    df_clean = df.copy()
    stats = []
    continuous_cols = detect_continuous_features(df)

    for col in continuous_cols:
        # 计算四分位数
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # 检测异常值
        outliers_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
        n_outliers = outliers_mask.sum()

        # 处理异常值
        if method == "remove":
            # 将异常值设为NaN
            df_clean[col] = np.where(outliers_mask, np.nan, df[col])
        elif method == "clip":
            df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)
        elif method == "median":
            median_val = df[col].median()
            df_clean.loc[outliers_mask, col] = median_val

        stats.append({
            "Feature": col,
            "Q1": Q1,
            "Q3": Q3,
            "IQR": IQR,
            "LowerBound": lower_bound,
            "UpperBound": upper_bound,
            "Outliers": n_outliers,
            "Method": method
        })

    stats_df = pd.DataFrame(stats)
    return df_clean, stats_df





def main():
    # 加载数据（排除ID和标签列）
    df = pd.read_excel(INPUT_PATH)
    id_label_cols = [df.columns[0]] + df.columns[-4:].tolist()
    features_df = df.drop(columns=id_label_cols)

    # 处理异常值
    df_clean, stats_df = process_outliers(features_df, method=OUTLIER_METHOD)

    # 合并ID和标签
    final_df = pd.concat([df[id_label_cols], df_clean], axis=1)

    # 保存结果
    final_df.to_excel(OUTPUT_PATH, index=False)
    stats_df.to_excel("outlier_stats.xlsx", index=False)
    print(f"异常值处理完成！结果保存至: {OUTPUT_PATH}")




if __name__ == "__main__":
    main()