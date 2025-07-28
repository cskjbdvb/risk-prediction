import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from fancyimpute import SoftImpute
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


# 1. 数据加载与预处理
def load_and_preprocess(filepath):
    df = pd.read_excel(filepath)

    # 分离ID、标签和特征
    id_col = df.iloc[:, 0]
    labels = df.iloc[:, 1:5]
    features = df.iloc[:, 5:]

    # 记录原始缺失情况
    missing_mask = features.isnull()
    missing_rates = missing_mask.mean().sort_values(ascending=False)

    print(f"原始数据缺失情况：\n{missing_rates.head(10)}")

    return id_col, labels, features, missing_mask


# 2. 评估函数
def evaluate_imputation(original, imputed, missing_mask):
    """
    评估指标说明：
    - MSE：均方误差，对较大误差更敏感
    - MAE：平均绝对误差，鲁棒性更强
    - Relative Error：相对于特征标准差的误差比例
    """
    # 仅计算原本缺失位置的误差
    original_values = original[missing_mask]
    imputed_values = imputed[missing_mask]

    mse = mean_squared_error(original_values, imputed_values)
    mae = mean_absolute_error(original_values, imputed_values)

    # 计算相对误差（与特征标准差比较）
    feature_stds = original.std()
    relative_errors = []
    for col in original.columns:
        col_mask = missing_mask[col]
        if col_mask.any():
            col_std = feature_stds[col]
            col_mse = mean_squared_error(
                original.loc[col_mask, col],
                imputed.loc[col_mask, col]
            )
            relative_errors.append(np.sqrt(col_mse) / col_std)

    avg_relative_error = np.mean(relative_errors)

    return {
        'MSE': mse,
        'MAE': mae,
        'Avg_Relative_Error': avg_relative_error,
        'Feature_Relative_Errors': relative_errors
    }


# 3. 参数调优函数



def tune_softimpute(features, missing_mask, param_grid):
    # 创建完整数据副本（仅用于评估）
    X_complete = features.copy()

    # 人为将完整数据设为已知缺失位置的平均值（模拟真实场景）
    X_missing = X_complete.mask(missing_mask)

    best_score = float('inf')
    best_params = None
    results = []

    # 兼容性工厂函数
    def create_imputer(r, mi):
        try:
            return SoftImpute(rank_k=r, max_iters=mi)
        except:
            try:
                return SoftImpute(rank=r, max_iters=mi)
            except:
                imp = SoftImpute()
                if hasattr(imp, 'rank'):
                    imp.rank = r
                elif hasattr(imp, 'rank_k'):
                    imp.rank_k = r
                imp.max_iters = mi
                return imp

    # 网格搜索
    for rank in tqdm(param_grid['rank_range'], desc="参数调优"):
        imputer = create_imputer(rank, param_grid.get('max_iters', 100))

        # 执行填充
        X_imputed = imputer.fit_transform(X_missing)
        X_imputed = pd.DataFrame(X_imputed, columns=features.columns)

        # 评估
        metrics = evaluate_imputation(X_complete, X_imputed, missing_mask)
        metrics['rank'] = rank
        results.append(metrics)

        if metrics['MSE'] < best_score:
            best_score = metrics['MSE']
            best_params = {'rank': rank}

    return pd.DataFrame(results), best_params

# 4. 可视化函数
def plot_results(results_df):
    plt.figure(figsize=(15, 5))

    # MSE随rank变化
    plt.subplot(1, 3, 1)
    sns.lineplot(data=results_df, x='rank', y='MSE')
    plt.title("MSE vs Rank")
    plt.xlabel("Matrix Rank")
    plt.ylabel("Mean Squared Error")

    # MAE随rank变化
    plt.subplot(1, 3, 2)
    sns.lineplot(data=results_df, x='rank', y='MAE')
    plt.title("MAE vs Rank")
    plt.xlabel("Matrix Rank")
    plt.ylabel("Mean Absolute Error")

    # 相对误差分布
    plt.subplot(1, 3, 3)
    all_errors = []
    for errors in results_df['Feature_Relative_Errors']:
        all_errors.extend(errors)
    sns.histplot(all_errors, bins=30, kde=True)
    plt.title("Relative Error Distribution")
    plt.xlabel("Error / Feature Std")

    plt.tight_layout()
    plt.show()


# 5. 主执行流程
def main(filepath):
    # 加载数据
    id_col, labels, features, missing_mask = load_and_preprocess(filepath)

    # 参数网格设置
    param_grid = {
        'rank_range': range(5, 51, 5),  # 测试5-50的rank，步长5
        'max_iters': 100
    }

    # 参数调优
    results_df, best_params = tune_softimpute(features, missing_mask, param_grid)

    # 可视化结果
    plot_results(results_df)

    print(f"\n最佳参数：{best_params}")
    print(f"最佳MSE：{results_df['MSE'].min():.4f}")
    print(f"最佳MAE：{results_df['MAE'].min():.4f}")
    print(f"平均相对误差：{results_df['Avg_Relative_Error'].min():.2%}")

    # 用最佳参数重新训练
    final_imputer = SoftImpute(
        rank_k=best_params['rank'],
        max_iters=param_grid['max_iters']
    )
    features_imputed = final_imputer.fit_transform(features)
    features_imputed = pd.DataFrame(features_imputed, columns=features.columns)

    # 保存结果
    final_df = pd.concat([id_col, labels, features_imputed], axis=1)
    final_df.to_excel("D:\\研二2\\论文撰写\\数据合并\\输出.xlsx", index=False)

    return final_df


if __name__ == "__main__":
    # 替换为您的Excel文件路径
    filepath = "D:\\研二2\\论文撰写\\数据合并\\独热编码、标准化.xlsx"
    final_df = main(filepath)