import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import hamming_loss, accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import ParameterGrid
import matplotlib.pyplot as plt
import matplotlib
from sklearn.multioutput import MultiOutputClassifier
import warnings

warnings.filterwarnings('ignore')


# ---------------------------
# 1. 数据加载与预处理
# ---------------------------
def load_data(train_path, test_path):
    """加载训练集和测试集"""
    train = pd.read_excel(train_path)
    test = pd.read_excel(test_path)

    # 分离特征和标签
    X_train = train.iloc[:, 1:-4]  # 跳过ID列，取中间特征列
    y_train = train.iloc[:, -4:]  # 最后4列是标签
    X_test = test.iloc[:, 1:-4]
    y_test = test.iloc[:, -4:]

    # 检查数据
    print(f"\n训练集形状: {X_train.shape}, 测试集形状: {X_test.shape}")
    print("标签分布（训练集）:\n", y_train.sum())

    return X_train, y_train, X_test, y_test


# ---------------------------
# 2. 参数网格设置
# ---------------------------
def get_param_grid():
    """定义待搜索的参数组合"""
    return {
        'boosting_type': ['gbdt', 'dart'],
        'num_leaves': [31, 63],
        'max_depth': [-1, 5],
        'learning_rate': [0.05, 0.1],
        'lambda_l1': [0, 0.1],
        'min_data_in_leaf': [20, 50],
        'feature_fraction': [0.8]
    }


# ---------------------------
# 3. 多标签评估函数
# ---------------------------

def evaluate(y_true, y_pred, y_prob):
    """计算多标签评估指标"""
    metrics = {
        'Hamming Loss': hamming_loss(y_true, y_pred),
        'Subset Accuracy': accuracy_score(y_true, y_pred),
        'Macro F1': f1_score(y_true, y_pred, average='macro'),
        'Micro F1': f1_score(y_true, y_pred, average='micro'),
        'ROC AUC OvR': roc_auc_score(y_true, y_prob, multi_class='ovr')
    }
    # 新增综合评分（可调整权重）
    weights = {
        'Hamming Loss': -0.3,  # 越小越好，故取负
        'Macro F1': 0.4,
        'Micro F1': 0.3
    }
    metrics['Composite Score'] = sum(metrics[k]*w for k,w in weights.items())
    return metrics

# ---------------------------
# 4. 特征选择主流程（修正多标签处理和可视化）
# ---------------------------
def lightgbm_feature_selection(train_path, test_path):
    # 加载数据
    X_train, y_train, X_test, y_test = load_data(train_path, test_path)

    # 参数搜索
    param_grid = get_param_grid()
    best_score = np.inf
    best_params = {}
    best_importance = None
    # 新增结果记录
    all_results = []

    print("\n开始参数搜索...")
    for params in ParameterGrid(param_grid):
        try:
            # 使用MultiOutputClassifier处理多标签
            model = MultiOutputClassifier(
                lgb.LGBMClassifier(
                    n_estimators=100,
                    random_state=42,
                    **params
                )
            )
            model.fit(X_train, y_train)

            # 预测
            y_pred = model.predict(X_test)
            y_prob = np.array([est.predict_proba(X_test)[:, 1] for est in model.estimators_]).T

            # 评估
            metrics = evaluate(y_test, y_pred, y_prob)
            metrics.update(params)  # 记录参数
            all_results.append(metrics)

            # 改用综合评分选择
            current_score = metrics['Composite Score']
            if current_score > best_score:  # 注意现在是越大越好
                best_score = current_score
                best_params = params
                best_importance = np.mean([est.feature_importances_ for est in model.estimators_], axis=0)
                print(f"发现更好参数: {params}")
                print(
                    f"评估指标: HammingLoss={metrics['Hamming Loss']:.4f}, MacroF1={metrics['Macro F1']:.4f}, MicroF1={metrics['Micro F1']:.4f}")

        except Exception as e:
            print(f"参数 {params} 失败: {str(e)}")
            continue

        # ---------------------------
        # 新增结果分析部分
        # ---------------------------
        # 保存所有结果
    results_df = pd.DataFrame(all_results)
    results_df.to_excel("all_parameter_results.xlsx", index=False)

    # 可视化指标关系
    plt.figure(figsize=(12, 8))
    plt.scatter(results_df['Macro F1'], results_df['Hamming Loss'],
                c=results_df['Composite Score'], s=100, alpha=0.6)
    plt.colorbar(label='Composite Score')
    plt.xlabel("Macro F1")
    plt.ylabel("Hamming Loss")
    plt.title("Parameter Performance Trade-off")
    plt.savefig('metrics_tradeoff.png', bbox_inches='tight', dpi=300)
    plt.close()



    # ---------------------------
    # 可视化（简化版，避免中文问题）
    # ---------------------------
    feature_importance = pd.Series(best_importance, index=X_train.columns)
    top_features = feature_importance.sort_values(ascending=False).head(30)

    plt.figure(figsize=(12, 8))
    top_features.plot(kind='barh')
    plt.title("Top 30 Important Features")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig('feature_importance.png', bbox_inches='tight', dpi=300)
    plt.close()

    print("\n特征重要性图已保存到 feature_importance.png")

    # 最终模型
    final_model = MultiOutputClassifier(
        lgb.LGBMClassifier(
            n_estimators=200,
            random_state=42,
            **best_params
        )
    )
    final_model.fit(X_train, y_train)

    y_pred = final_model.predict(X_test)
    y_prob = np.array([est.predict_proba(X_test)[:, 1] for est in final_model.estimators_]).T
    final_metrics = evaluate(y_test, y_pred, y_prob)

    print("\n最终评估结果:")
    for metric, value in final_metrics.items():
        print(f"{metric}: {value:.4f}")

    selected_features = feature_importance[feature_importance > 0].index.tolist()
    return selected_features, final_metrics


# ---------------------------
# 执行代码
# ---------------------------
if __name__ == "__main__":
    train_path = "D:\\研二2\\论文撰写\\数据合并\\标准化、缺失值\\train_resampled_manual_mlsmote.xlsx"  # 替换为您的训练集路径
    test_path = "D:\\研二2\\论文撰写\\数据合并\\标准化、缺失值\\测试集.xlsx"  # 替换为您的测试集路径

    selected_features, metrics = lightgbm_feature_selection(train_path, test_path)
    pd.Series(selected_features).to_excel("selected_features.xlsx", index=False)
    print("\n特征选择完成！结果已保存到 selected_features.xlsx")


