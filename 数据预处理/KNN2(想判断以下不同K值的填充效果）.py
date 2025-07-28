import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.base import TransformerMixin
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.utils.validation import check_is_fitted
import warnings

warnings.filterwarnings('ignore')

######################################################
#                  用户配置区域                      #
######################################################
INPUT_PATH = "D:\\研二2\\论文撰写\\数据合并\\箱线图.xlsx"  # 输入数据路径
OUTPUT_PATH = "D:\\研二2\\论文撰写\\数据合并\\KNN填充缺失值.xlsx"  # 输出路径
KNN_NEIGHBORS = 6  # KNN近邻数
CATEGORY_THRESHOLD = 10  # 唯一值≤此值为分类特征
MISSING_ONLY = True  # 是否仅填充有缺失的列
SAMPLE_FRACTION = 0.1  # 评估抽样比例


######################################################

class FeatureKNNImputer(TransformerMixin):
    """专门处理特征列的KNN填充器"""

    def __init__(self, n_neighbors=5, cat_threshold=10):
        self.n_neighbors = n_neighbors
        self.cat_threshold = cat_threshold
        self.num_cols = []
        self.cat_cols = []
        self.scaler = StandardScaler()
        self.imputer = KNNImputer(n_neighbors=n_neighbors)
        self.cat_encoders = {}
        self.cat_modes = {}

    def _detect_feature_types(self, X):
        """自动检测特征类型"""
        self.num_cols, self.cat_cols = [], []

        for col in X.columns:
            if X[col].isnull().all():  # 跳过全缺失列
                continue

            unique_vals = X[col].dropna().nunique()
            if pd.api.types.is_numeric_dtype(X[col]):
                if unique_vals <= self.cat_threshold:
                    self.cat_cols.append(col)
                else:
                    self.num_cols.append(col)
            else:
                self.cat_cols.append(col)

    def fit(self, X, y=None):
        self._detect_feature_types(X)

        # 预处理分类特征
        for col in self.cat_cols:
            le = LabelEncoder()
            non_null = X[col].dropna()
            if len(non_null) > 0:
                le.fit(non_null.astype(str))
                self.cat_encoders[col] = le
                self.cat_modes[col] = le.transform([non_null.mode()[0]])[0]

        # 预处理连续型特征
        if self.num_cols:
            num_data = X[self.num_cols].select_dtypes(include=np.number)
            if not num_data.empty:
                self.scaler.fit(num_data)
        return self

    def transform(self, X):
        check_is_fitted(self)
        X_trans = X.copy()

        # 处理连续型特征
        if self.num_cols:
            num_data = X_trans[self.num_cols].select_dtypes(include=np.number)
            if not num_data.empty:
                scaled = self.scaler.transform(num_data)
                filled = self.imputer.fit_transform(scaled)
                X_trans[self.num_cols] = self.scaler.inverse_transform(filled)

        # 处理分类特征
        for col in self.cat_cols:
            if col not in self.cat_encoders:
                continue

            le = self.cat_encoders[col]

            # 编码已知值
            known_mask = X_trans[col].notna()
            X_trans.loc[known_mask, col] = le.transform(X_trans.loc[known_mask, col].astype(str))

            # 填充缺失值
            col_data = X_trans[col].to_numpy().reshape(-1, 1)
            filled = self.imputer.fit_transform(col_data)
            filled_ints = np.round(filled).astype(int).flatten()

            # 确保填充值在有效范围内
            valid_mask = np.isin(filled_ints, le.transform(le.classes_))
            filled_ints[~valid_mask] = self.cat_modes[col]
            X_trans[col] = filled_ints

            # 解码回原始标签
            X_trans[col] = le.inverse_transform(X_trans[col].astype(int))

        return X_trans


def evaluate_imputation(X_features, n_neighbors, cat_threshold, sample_frac):
    """评估特征列填充效果"""
    X_eval = X_features.copy()
    true_values = {}
    masks = {}

    # 创建模拟缺失值
    for col in X_eval.columns:
        non_missing = X_eval[col].dropna().index
        if len(non_missing) > 10:  # 至少需要10个样本
            sample_size = max(1, int(len(non_missing) * sample_frac))
            sampled_idx = np.random.choice(non_missing, size=sample_size, replace=False)
            true_values[col] = X_eval.loc[sampled_idx, col].copy()
            masks[col] = sampled_idx
            X_eval.loc[sampled_idx, col] = np.nan

    # 填充并评估
    imputer = FeatureKNNImputer(n_neighbors=n_neighbors, cat_threshold=cat_threshold)
    X_filled = imputer.fit_transform(X_eval)

    # 计算评估指标
    mse_list, acc_list = [], []
    for col in masks:
        idx = masks[col]
        true = true_values[col]
        pred = X_filled.loc[idx, col]

        if col in imputer.num_cols:
            mse = mean_squared_error(true, pred)
            mse_list.append(mse)
        elif col in imputer.cat_cols:
            # 分类特征比较原始字符串值
            acc = accuracy_score(true.astype(str), pred.astype(str))
            acc_list.append(acc)

    # 打印结果
    print("\n" + "=" * 50)
    print(f"评估结果（K={n_neighbors}）:")
    if mse_list:
        print(f"连续型特征 MSE: 平均{np.mean(mse_list):.4f} 范围[{np.min(mse_list):.4f}, {np.max(mse_list):.4f}]")
    if acc_list:
        print(f"分类型特征准确率: 平均{np.mean(acc_list):.2%} 范围[{np.min(acc_list):.2%}, {np.max(acc_list):.2%}]")
    print("=" * 50)


def main():
    # 1. 加载数据
    try:
        df = pd.read_excel(INPUT_PATH)
        print(f"原始数据维度: {df.shape}")
        print("初始缺失值统计:")
        print(df.isnull().sum().sort_values(ascending=False).head(10))
    except Exception as e:
        print(f"加载数据失败: {e}")
        return

    # 2. 分离数据
    id_col = df.columns[0]  # 第一列是ID
    label_cols = df.columns[-4:]  # 最后四列是标签
    feature_cols = df.columns[1:-4]  # 中间所有列是特征

    df_ids = df[[id_col]]
    df_labels = df[label_cols]
    df_features = df[feature_cols]

    # 3. 评估特征列填充效果
    print("\n正在评估填充效果...")
    evaluate_imputation(
        X_features=df_features.copy(),
        n_neighbors=KNN_NEIGHBORS,
        cat_threshold=CATEGORY_THRESHOLD,
        sample_frac=SAMPLE_FRACTION
    )

    # 4. 实际填充特征列
    print("\n正在进行缺失值填充...")
    imputer = FeatureKNNImputer(
        n_neighbors=KNN_NEIGHBORS,
        cat_threshold=CATEGORY_THRESHOLD
    )

    if MISSING_ONLY:
        missing_cols = df_features.columns[df_features.isnull().any()].tolist()
        if missing_cols:
            print(f"需要填充的列: {missing_cols}")
            df_filled = imputer.fit_transform(df_features[missing_cols])
            df_features.update(df_filled)
    else:
        df_features = imputer.fit_transform(df_features)

    # 5. 合并保存结果
    final_df = pd.concat([df_ids, df_features, df_labels], axis=1)
    try:
        final_df.to_excel(OUTPUT_PATH, index=False)
        print(f"\n预处理完成！保存至: {OUTPUT_PATH}")
        print("处理后数据维度:", final_df.shape)
        print("最终缺失值统计:")
        print(final_df.isnull().sum().sort_values(ascending=False).head(10))
    except Exception as e:
        print(f"保存结果失败: {e}")


if __name__ == "__main__":
    main()