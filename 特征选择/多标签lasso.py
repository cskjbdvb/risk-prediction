import pandas as pd
import numpy as np
from sklearn.linear_model import MultiTaskLassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import hamming_loss, f1_score, roc_auc_score
import matplotlib.pyplot as plt


# ---------------------------
# 1. 数据加载与预处理
# ---------------------------
def load_data(train_path, test_path):
    train = pd.read_excel(train_path)
    test = pd.read_excel(test_path)

    # 分离特征和标签
    X_train = train.iloc[:, 1:-4]  
    y_train = train.iloc[:, -4:] 
    X_test = test.iloc[:, 1:-4]
    y_test = test.iloc[:, -4:]

    # 获取特征名称
    feature_names = X_train.columns.tolist()

    # 数据标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"\n训练集形状: {X_train.shape}, 测试集形状: {X_test.shape}")
    print("标签分布:\n", y_train.sum())

    return X_train, y_train, X_test, y_test, feature_names


# ---------------------------
# 2. 多标签评估函数
def evaluate(y_true, y_pred, y_scores):
  
    metrics = {
        'Hamming Loss': hamming_loss(y_true, y_pred),
        'Macro F1': f1_score(y_true, y_pred, average='macro'),
        'Micro F1': f1_score(y_true, y_pred, average='micro'),
        'ROC AUC OvR': roc_auc_score(y_true, y_scores, multi_class='ovr')
    }
    return metrics


# ---------------------------
# 3. 多标签Lasso特征选择
# ---------------------------



def multitask_lasso_feature_selection(train_path, test_path):
    """
    多标签Lasso特征选择主流程
    """
    # 加载数据
    X_train, y_train, X_test, y_test, feature_names = load_data(train_path, test_path)

    # 模型训练
    alphas = np.logspace(-4, 0, 50)
    model = MultiTaskLassoCV(
        alphas=alphas,
        cv=5,
        n_jobs=-1,
        random_state=42,
        selection='cyclic'
    )
    model.fit(X_train, y_train)

    print(f"\n最佳alpha值: {model.alpha_:.6f}")

    # 特征选择
    coef_matrix = model.coef_.T  # 转置为(115,4)
    selected_mask = np.any(coef_matrix != 0, axis=1)
    selected_features = [feature_names[i] for i in np.where(selected_mask)[0]]

    print(f"\n从 {len(feature_names)} 个特征中选出了 {len(selected_features)} 个重要特征")

    # 模型评估
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    y_scores = model.predict(X_test)
    metrics = evaluate(y_test, y_pred, y_scores)

    print("\n评估结果:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    # 结果可视化
    coef_df = pd.DataFrame(
        coef_matrix,
        columns=y_test.columns,
        index=feature_names
    )

    # 非零系数可视化
    nonzero_coef = coef_matrix[selected_mask]
    nonzero_features = selected_features

    plt.figure(figsize=(12, 8))
    plt.imshow(nonzero_coef, cmap='coolwarm', aspect='auto')
    plt.colorbar(label='系数值')
    plt.yticks(range(len(nonzero_features)), nonzero_features)
    plt.xticks(range(4), y_test.columns)
    plt.xlabel("疾病标签")
    plt.ylabel("特征")
    plt.title("多标签Lasso系数热力图 (非零系数)")
    plt.tight_layout()
    plt.savefig('lasso_coefficients_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

    return selected_features, metrics, coef_df

# ---------------------------
# 4. 执行代码
# ---------------------------
if __name__ == "__main__":
    # 文件路径（替换为您的实际路径）
    train_path = "标准化.xlsx"
    test_path = "测试.xlsx"

    # 运行特征选择
    selected_features, metrics, coef_df = multitask_lasso_feature_selection(train_path, test_path)

    # 保存结果
    pd.Series(selected_features).to_excel("selected_features_lasso.xlsx", index=False)
    coef_df.to_excel("lasso_coefficients.xlsx")

    print("\n特征选择完成！")
    print("选中的特征已保存到 selected_features_lasso.xlsx")
    print("系数矩阵已保存到 lasso_coefficients.xlsx")
