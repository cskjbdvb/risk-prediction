import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


def load_data(filepath):
    """加载数据并分离特征"""
    df = pd.read_excel(filepath)
    print(f"数据加载成功，形状: {df.shape}")

    # 第1列ID，2-5列标签，其余特征
    id_col = df.iloc[:, 0]
    labels = df.iloc[:, 1:5]
    features = df.iloc[:, 5:]

    print("\n缺失值统计:")
    print(features.isnull().sum().sort_values(ascending=False).head())
    return id_col, labels, features


def prepare_validation(features, sample_frac=0.2):
    """准备验证集"""
    # 取部分完整数据作为验证基准
    complete_mask = ~features.isnull().any(axis=1)
    if complete_mask.sum() > 100:
        val_data = features[complete_mask].sample(frac=sample_frac)
    else:
        print("警告：完整数据不足，使用部分填充数据")
        val_data = features.sample(frac=sample_frac)

    # 人为制造5%缺失
    np.random.seed(42)
    mask = np.random.rand(*val_data.shape) < 0.05
    val_missing = val_data.mask(mask)
    return val_data, val_missing, mask


def tune_mice(features, n_trials=5):
    """简化参数调优"""
    val_true, val_missing, mask = prepare_validation(features)

    best_score = float('inf')
    best_params = {}

    # 测试不同参数组合
    for max_iter in [10, 20, 30]:
        imputer = IterativeImputer(
            estimator=BayesianRidge(),
            max_iter=max_iter,
            random_state=42
        )

        try:
            imputed = imputer.fit_transform(val_missing)
            score = mean_squared_error(val_true[mask], imputed[mask])
            if score < best_score:
                best_score = score
                best_params = {'max_iter': max_iter}
        except:
            continue

    print(f"\n最佳参数: {best_params}, MSE: {best_score:.4f}")
    return best_params


def run_mice_imputation(filepath):
    """完整执行流程"""
    # 1. 加载数据
    id_col, labels, features = load_data(filepath)

    # 2. 参数调优
    best_params = tune_mice(features)

    # 3. 执行填补
    imputer = IterativeImputer(
        estimator=BayesianRidge(),
        max_iter=best_params['max_iter'],
        random_state=42
    )
    features_imputed = imputer.fit_transform(features)
    features_imputed = pd.DataFrame(features_imputed, columns=features.columns)

    # 4. 保存结果
    final_df = pd.concat([id_col, labels, features_imputed], axis=1)
    final_df.to_excel("D:\\研二2\\论文撰写\\数据合并\\输出.xlsx", index=False)

    # 5. 验证结果
    print("\n填补后缺失值检查:", features_imputed.isnull().sum().sum())

    # 可视化示例特征
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(features.iloc[:, 0].dropna(), bins=30, alpha=0.7, label='Original')
    plt.title("原始分布")

    plt.subplot(1, 2, 2)
    plt.hist(features_imputed.iloc[:, 0], bins=30, alpha=0.7, color='orange', label='Imputed')
    plt.title("填补后分布")
    plt.tight_layout()
    plt.show()

    return final_df


# 直接运行（修改为您的文件路径）
if __name__ == "__main__":
    final_result = run_mice_imputation("D:\\研二2\\论文撰写\\数据合并\\独热编码、标准化.xlsx")
    print("处理完成！结果已保存到 mice_imputed_result.xlsx")