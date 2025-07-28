# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from collections import Counter
from imblearn.over_sampling import SMOTE

######################################################
#                  用户配置区域                      #
######################################################
INPUT_TRAIN_PATH = "D:\\研二2\\论文撰写\\训练集.xlsx"  # 输入标准化训练集路径
OUTPUT_TRAIN_PATH = "D:\\研二2\\论文撰写\\对训练集进行ML-SMOTE.xlsx"  # 输出过采样后训练集路径
MINORITY_THRESHOLD = 0.1  # 定义少数类阈值（正样本比例<10%视为少数类）
K_NEIGHBORS = 3  # SMOTE近邻数
RANDOM_STATE = 42  # 随机种子


######################################################

class AutoMLSMOTE:
    def __init__(self, minority_threshold=0.1, k_neighbors=5, random_state=42):
        self.minority_threshold = minority_threshold
        self.k_neighbors = k_neighbors
        self.random_state = random_state

    def _auto_detect_minority_labels(self, y):
        """自动识别需要处理的少数类标签"""
        n_samples = y.shape[0]
        label_stats = []

        for idx in range(y.shape[1]):
            positive_count = sum(y[:, idx])
            ratio = positive_count / n_samples
            label_stats.append((
                idx,
                f"Label_{idx + 1}",
                positive_count,
                ratio,
                ratio < self.minority_threshold
            ))

        print("\n标签分析报告：")
        print(f"{'标签':<8} {'正样本数':<10} {'比例':<10} {'是否少数类':<10}")
        for stat in label_stats:
            print(f"{stat[1]:<8} {stat[2]:<10} {stat[3]:<.4f}    {str(stat[4]):<10}")

        return [stat[0] for stat in label_stats if stat[4]]

    def _generate_sampling_strategy(self, y, minority_labels):
        """生成动态采样策略"""
        sampling_strategy = {}
        original_combined = Counter(['-'.join(map(str, row)) for row in y])

        for label_idx in minority_labels:
            for combo, count in original_combined.items():
                labels = combo.split('-')
                if labels[label_idx] == '1':
                    target_count = min(count * 3, int(len(y) * 0.3))  # 最大增加到30%比例
                    sampling_strategy[combo] = target_count
        return sampling_strategy

    def fit_resample(self, X, y):
        """执行智能过采样"""
        # 自动检测少数类标签
        minority_labels = self._auto_detect_minority_labels(y)
        if not minority_labels:
            print("未检测到需要处理的少数类标签")
            return X, y

        # 生成组合标签
        y_combined = ['-'.join(map(str, row)) for row in y.astype(int)]

        # 动态生成采样策略
        sampling_strategy = self._generate_sampling_strategy(y, minority_labels)
        print("\n采用的采样策略：", sampling_strategy)

        # 执行多标签SMOTE
        smote = SMOTE(
            sampling_strategy=sampling_strategy,
            k_neighbors=self.k_neighbors,
            random_state=self.random_state
        )
        X_res, y_combined_res = smote.fit_resample(X, y_combined)

        # 转换回多标签格式
        y_res = np.array([[int(l) for l in s.split('-')] for s in y_combined_res])

        # 生成新ID
        new_ids = [f"org_{i}" for i in range(len(y))] + \
                  [f"syn_{i}" for i in range(len(y_res) - len(y))]

        return X_res, y_res, new_ids


def main():
    # 读取数据
    df = pd.read_excel(INPUT_TRAIN_PATH)
    print(f"原始数据维度: {df.shape}")

    # 分离数据
    feature_cols = df.columns[1:-4]  # 跳过ID列和最后4列标签
    X = df[feature_cols].values
    y = df.iloc[:, -4:].values

    # 执行智能过采样
    processor = AutoMLSMOTE(
        minority_threshold=MINORITY_THRESHOLD,
        k_neighbors=K_NEIGHBORS,
        random_state=RANDOM_STATE
    )
    X_res, y_res, new_ids = processor.fit_resample(X, y)

    # 构建结果DataFrame
    df_res = pd.DataFrame(X_res, columns=feature_cols)
    for idx in range(y.shape[1]):
        df_res[f"Label_{idx + 1}"] = y_res[:, idx]
    df_res.insert(0, df.columns[0], new_ids)  # 插入ID列

    # 保存结果
    df_res.to_excel(OUTPUT_TRAIN_PATH, index=False)
    print(f"\n结果已保存至: {OUTPUT_TRAIN_PATH}")
    print(f"新数据维度: {df_res.shape}")
    print("新数据标签分布:")
    for idx in range(y.shape[1]):
        print(f"Label_{idx + 1}: {sum(y_res[:, idx])}")


if __name__ == "__main__":
    main()