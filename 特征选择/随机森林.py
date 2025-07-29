import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import hamming_loss, accuracy_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')


# ---------------------------
# 1. 数据加载

def load_data(train_path, test_path):
    train = pd.read_excel(train_path)
    test = pd.read_excel(test_path)

    # 分离特征和标签
    X_train = train.iloc[:, 1:-4].values  # 特征矩阵
    y_train = train.iloc[:, -4:].values  # 多标签矩阵
    X_test = test.iloc[:, 1:-4].values
    y_test = test.iloc[:, -4:].values

    # 获取特征名称
    feature_names = train.columns[1:-4].tolist()

    print(f"\n训练集形状: {X_train.shape}, 测试集形状: {X_test.shape}")
    print("标签分布:\n", pd.DataFrame(y_train, columns=train.columns[-4:]).sum())

    return X_train, y_train, X_test, y_test, feature_names


# ---------------------------
# 2. 多标签评估函数

def evaluate(y_true, y_pred, y_prob):
    """计算多标签评估指标"""
    metrics = {
        'Hamming Loss': hamming_loss(y_true, y_pred),
        'Subset Accuracy': accuracy_score(y_true, y_pred),
        'Macro F1': f1_score(y_true, y_pred, average='macro'),
        'Micro F1': f1_score(y_true, y_pred, average='micro'),
        'ROC AUC OvR': roc_auc_score(y_true, y_prob, multi_class='ovr')
    }
    # 综合评分（可调整权重）
    metrics['Composite Score'] = (0.4 * metrics['Macro F1'] +
                                  0.3 * metrics['Micro F1'] +
                                  0.3 * (1 - metrics['Hamming Loss']))
    return metrics


# ---------------------------
# 3. 参数网格设置
def get_param_grid():
    """定义随机森林参数组合"""
    return {
        'n_estimators': [100, 200],  # 树的数量
        'max_depth': [None, 10, 20],  # 树的最大深度
        'min_samples_split': [2, 5],  # 分裂所需最小样本数
        'min_samples_leaf': [1, 3],  # 叶节点最小样本数
        'max_features': ['sqrt', 0.8],  # 考虑的最大特征比例
        'bootstrap': [True, False]  # 是否使用bootstrap采样
    }


# ---------------------------
# 4. 多标签随机森林特征选择
# ---------------------------
def rf_feature_selection(train_path, test_path):
    # 加载数据
    X_train, y_train, X_test, y_test, feature_names = load_data(train_path, test_path)

    # 参数搜索
    param_grid = get_param_grid()
    all_results = []
    best_score = -np.inf
    best_model = None

    print("\n开始参数搜索...")
    for params in ParameterGrid(param_grid):
        try:
            # 多标签随机森林
            rf = RandomForestClassifier(
                random_state=42,
                n_jobs=-1,  # 使用所有CPU核心
                **params
            )
            model = MultiOutputClassifier(rf)

            # 训练
            model.fit(X_train, y_train)

            # 预测
            y_pred = model.predict(X_test)
            y_prob = np.array([est.predict_proba(X_test)[:, 1] for est in model.estimators_]).T

            # 评估
            metrics = evaluate(y_test, y_pred, y_prob)
            metrics.update(params)  # 记录参数
            all_results.append(metrics)

            print(f"\n参数组合: {params}")
            print(f"Hamming Loss: {metrics['Hamming Loss']:.4f}")
            print(f"Macro F1: {metrics['Macro F1']:.4f}")
            print(f"Composite Score: {metrics['Composite Score']:.4f}")

            # 更新最佳模型
            if metrics['Composite Score'] > best_score:
                best_score = metrics['Composite Score']
                best_model = model
                best_params = params

        except Exception as e:
            print(f"参数 {params} 失败: {str(e)}")
            continue

    # ---------------------------
    # 结果分析
    # ---------------------------
    # 保存所有结果
    results_df = pd.DataFrame(all_results)
    results_df.to_excel("rf_parameter_results.xlsx", index=False)

    # 特征重要性分析
    importances = np.mean([est.feature_importances_ for est in best_model.estimators_], axis=0)
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)

    # 可视化Top30特征
    plt.figure(figsize=(12, 8))
    importance_df.head(30).plot.barh(x='Feature', y='Importance')
    plt.title("Top 30 Important Features (Random Forest)")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig('rf_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 保存重要特征
    selected_features = importance_df[importance_df['Importance'] > 0]['Feature'].tolist()
    importance_df.to_excel("rf_feature_importances.xlsx", index=False)

    print("\n最佳参数组合:", best_params)
    print(f"选出了 {len(selected_features)} 个重要特征")

    return selected_features, best_params, results_df


# ---------------------------
# 5. 执行代码
# ---------------------------
if __name__ == "__main__":
    # 文件路径配置
    train_path = "标准化.xlsx"
    test_path = "测试.xlsx"

    # 运行特征选择
    selected_features, best_params, results_df = rf_feature_selection(train_path, test_path)

    # 保存结果
    pd.Series(selected_features).to_excel("selected_features_rf.xlsx", index=False)
    print("\n特征选择完成！")
    print(f"最佳参数: {best_params}")
    print(f"选中的特征已保存到 selected_features_rf.xlsx")
    print(f"所有参数结果已保存到 rf_parameter_results.xlsx")
