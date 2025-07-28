import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# ===================== 参数配置 =====================
INPUT_FILE = "D:\\研二2\\论文撰写\\数据合并\\标准化.xlsx"  # 替换为你的数据文件路径
SHEET_NAME = "填充缺失值"  # 工作表名称
ID_COL = 'SEQN'  # ID列名
LABEL_COLS = ['DIQ010 - 医生告诉你有糖尿病', 'MCQ160c - 曾被告知自己患有冠心病', 'MCQ160f - 曾被告知你中风','是否被告知患有COPD、肺气肿或慢性支气管炎']  # 四个标签列名
K_VALUES = [15,17,19,21,23]  # 要测试的K值范围
TEST_SIZE = 0.2  # 用于评估的测试集比例
RANDOM_STATE = 42  # 随机种子


# ==================================================

def load_data():
    """加载数据"""
    df = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME)
    print("\n=== 数据加载成功 ===")
    print(f"数据形状: {df.shape}")
    print("\n前3行数据预览:")
    print(df.head(3))
    return df


def prepare_data(df):
    """准备数据：分离特征和标签"""
    # 分离ID和标签
    ids = df[ID_COL]
    labels = df[LABEL_COLS]

    # 特征列（所有非ID非标签列）
    features = df.drop(columns=[ID_COL] + LABEL_COLS)

    # 找出有缺失值的列
    missing_cols = features.columns[features.isna().any()].tolist()
    complete_cols = [col for col in features.columns if col not in missing_cols]

    print(f"\n有缺失值的特征列 ({len(missing_cols)}个): {missing_cols}")
    print(f"完整无缺的特征列 ({len(complete_cols)}个): {complete_cols}")

    return features, missing_cols, complete_cols, ids, labels


def evaluate_imputation(original, imputed, mask):
    """评估填充效果"""
    # 获取有效值索引
    valid_mask = mask & ~np.isnan(original) & ~np.isnan(imputed)

    if np.sum(valid_mask) == 0:
        return np.nan, np.nan

    # 只计算有效值
    mse = mean_squared_error(original[valid_mask], imputed[valid_mask])
    mae = mean_absolute_error(original[valid_mask], imputed[valid_mask])
    return mse, mae


def knn_impute_with_evaluation(features, missing_cols, complete_cols, k_values):
    """KNN填充 + 评估（只处理有缺失值的列）"""
    # 复制数据用于评估
    X = features.copy()

    # 1. 找出所有有缺失值的行和列
    missing_mask = pd.DataFrame(np.isnan(X[missing_cols]),
                                index=X.index,
                                columns=missing_cols)

    # 2. 人为制造更多缺失值用于评估
    evaluation_mask = pd.DataFrame(np.zeros_like(X[missing_cols], dtype=bool),
                                   index=X.index,
                                   columns=missing_cols)

    for col in missing_cols:
        # 找出该列非缺失的索引
        non_missing_idx = X[col].dropna().index
        # 从中随机选择20%作为评估用缺失值
        eval_idx = np.random.choice(non_missing_idx,
                                    size=int(len(non_missing_idx) * TEST_SIZE),
                                    replace=False)
        evaluation_mask.loc[eval_idx, col] = True

    # 合并原始缺失和评估缺失
    total_mask = missing_mask | evaluation_mask

    # 备份真实值（仅缺失列）
    X_true = X[missing_cols].copy()
    X_missing = X.copy()
    X_missing[total_mask] = np.nan

    # 标准化数据（仅对缺失列）
    scaler = StandardScaler()
    X_scaled_missing = pd.DataFrame(scaler.fit_transform(X_missing[missing_cols]),
                                    columns=missing_cols,
                                    index=X_missing.index)

    # 合并完整列（不处理）
    X_scaled = pd.concat([X[complete_cols], X_scaled_missing], axis=1)

    # 网格搜索最佳K值
    results = []
    best_k = None
    best_score = float('inf')

    print("\n=== 开始KNN填充评估 ===")
    for k in tqdm(k_values, desc="评估不同K值"):
        try:
            # 使用KNN填充（仅处理缺失列）
            imputer = KNNImputer(n_neighbors=k)
            X_imputed_missing = imputer.fit_transform(X_scaled[missing_cols])
            X_imputed_missing = pd.DataFrame(scaler.inverse_transform(X_imputed_missing),
                                             columns=missing_cols,
                                             index=X_missing.index)

            # 合并回完整列
            X_imputed = pd.concat([X[complete_cols], X_imputed_missing], axis=1)

            # 评估填充效果（仅缺失列）
            mse, mae = evaluate_imputation(X_true.values,
                                           X_imputed[missing_cols].values,
                                           evaluation_mask.values)

            if not np.isnan(mse):  # 忽略无效评估
                results.append({'k': k, 'mse': mse, 'mae': mae})

                # 选择MSE最小的K值
                if mse < best_score:
                    best_score = mse
                    best_k = k
        except Exception as e:
            print(f"K={k}时出错: {str(e)}")
            continue

    # 输出评估结果
    results_df = pd.DataFrame(results)
    print("\n=== K值评估结果 ===")
    print(results_df)
    print(f"\n最佳K值: {best_k} (最低MSE: {best_score:.4f})")

    # 用最佳K值填充原始数据的所有缺失值（仅缺失列）
    print("\n=== 使用最佳K值进行最终填充 ===")
    final_imputer = KNNImputer(n_neighbors=best_k)
    X_final_missing = pd.DataFrame(scaler.fit_transform(features[missing_cols]),
                                   columns=missing_cols,
                                   index=features.index)
    X_final_missing = final_imputer.fit_transform(X_final_missing)
    X_final_missing = pd.DataFrame(scaler.inverse_transform(X_final_missing),
                                   columns=missing_cols,
                                   index=features.index)

    # 合并回完整列
    X_final = pd.concat([features[complete_cols], X_final_missing], axis=1)

    return X_final, best_k, results_df


def save_results(final_df, ids, labels, best_k, results_df):
    """保存结果"""
    # 合并回ID和标签列
    final_df[ID_COL] = ids.values
    for label in LABEL_COLS:
        final_df[label] = labels[label].values

    # 调整列顺序
    cols = [ID_COL] + LABEL_COLS + [col for col in final_df.columns if col not in [ID_COL] + LABEL_COLS]
    final_df = final_df[cols]

    # 保存填充后的数据
    output_file = "D:\\研二2\\论文撰写\\数据合并\\KNN.xlsx"
    final_df.to_excel(output_file, index=False)

    # 保存评估结果
    eval_file = "D:\\研二2\\论文撰写\\数据合并\\KNN系数.xlsx"
    results_df.to_excel(eval_file, index=False)

    print("\n=== 处理完成 ===")
    print(f"✅ 填充后的数据已保存到: {output_file}")
    print(f"✅ 评估结果已保存到: {eval_file}")
    print(f"🔄 使用的K值: {best_k}")
    print("\n填充后数据预览:")
    print(final_df.head(3))


if __name__ == "__main__":
    # 1. 加载数据
    df = load_data()

    # 2. 准备数据
    features, missing_cols, complete_cols, ids, labels = prepare_data(df)

    # 3. KNN填充 + 评估（只处理缺失列）
    final_features, best_k, results_df = knn_impute_with_evaluation(
        features, missing_cols, complete_cols, K_VALUES)

    # 4. 保存结果
    save_results(final_features, ids, labels, best_k, results_df)