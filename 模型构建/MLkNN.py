import pandas as pd
import numpy as np
from skmultilearn.adapt import MLkNN
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.metrics import (
    hamming_loss, accuracy_score,
    f1_score, roc_auc_score,
    precision_score, recall_score,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------
# 1. 数据加载与预处理
# ---------------------------
def load_and_split_data(data_path, test_size=0.2, random_state=42):
    data = pd.read_excel(data_path)

    # 分离特征和标签
    X = data.iloc[:, 1:-4].values
    y = data.iloc[:, -4:].values.astype(int)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 获取标签名称
    label_names = data.columns[-4:].tolist()

    print(f"\n训练集形状: {X_train.shape}, 测试集形状: {X_test.shape}")
    print("训练集标签分布:\n", pd.DataFrame(y_train, columns=label_names).sum())
    print("测试集标签分布:\n", pd.DataFrame(y_test, columns=label_names).sum())

    return X_train, y_train, X_test, y_test, label_names


# ---------------------------
# 2. 多标签评估函数
# ---------------------------
def evaluate(y_true, y_pred, y_scores=None):
    """计算多标签评估指标"""
    metrics = {
        'Hamming Loss': hamming_loss(y_true, y_pred),
        'Subset Accuracy': accuracy_score(y_true, y_pred),
        'Macro Precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'Macro Recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'Macro F1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'Micro F1': f1_score(y_true, y_pred, average='micro', zero_division=0),
    }

    # 如果提供预测概率则计算AUC
    if y_scores is not None:
        try:
            metrics['ROC AUC OvR'] = roc_auc_score(y_true, y_scores, average='macro', multi_class='ovr')
        except:
            metrics['ROC AUC OvR'] = np.nan

    # 综合评分（可调整权重）
    weights = {
        'Macro F1': 0.4,
        'Micro F1': 0.3,
        'Hamming Loss': -0.3  # 越小越好，取负
    }
    metrics['Composite Score'] = sum(metrics[k] * w for k, w in weights.items())

    return metrics


def mlknn_modeling(data_path):
    # 加载并划分数据
    X_train, y_train, X_test, y_test, label_names = load_and_split_data(data_path)

    # 参数网格设置
    param_grid = {
        'k': [5, 10, 15],  # 最近邻数量
        's': [0.5, 1.0, 1.5],  # 平滑参数
    }

    # 存储所有结果
    all_results = []
    best_score = -np.inf
    best_model = None
    best_params = {'k': 10, 's': 1.0}  # 默认值

    print("\n开始MLkNN参数搜索...")
    for params in ParameterGrid(param_grid):
        try:
            # 修正的MLkNN初始化方式
            model = MLkNN(
                k=params['k'],
                s=params['s']
            )

            # 训练模型
            model.fit(X_train, y_train)

            # 预测
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None

            # 评估
            metrics = evaluate(y_test, y_pred.toarray(), y_prob.toarray() if y_prob else None)
            metrics.update(params)  # 记录当前参数
            all_results.append(metrics)

            print(f"\n参数组合: {params}")
            for name, val in metrics.items():
                if name not in params:
                    print(f"{name}: {val:.4f}")

            # 更新最佳模型
            if metrics['Composite Score'] > best_score:
                best_score = metrics['Composite Score']
                best_model = model
                best_params = params

        except Exception as e:
            print(f"参数 {params} 失败: {str(e)}")
            continue

    if best_model is None:
        print("\n所有参数组合失败，使用默认参数 k=10, s=1.0")
        best_model = MLkNN(k=10, s=1.0)
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test) if hasattr(best_model, 'predict_proba') else None
        metrics = evaluate(y_test, y_pred.toarray(), y_prob.toarray() if y_prob else None)
        all_results.append(metrics)
        best_params = {'k': 10, 's': 1.0}

    # 结果分析与可视化

    results_df = pd.DataFrame(all_results)
    results_df.to_excel("mlknn_parameter_results22.xlsx", index=False)

    # 最佳模型评估
    print("\n最佳参数组合:", best_params)
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test) if hasattr(best_model, 'predict_proba') else None

    # 详细分类报告
    print("\n分类报告:")
    print(classification_report(y_test, y_pred.toarray(), target_names=label_names, zero_division=0))

    # 各标签性能热力图
    label_metrics = []
    for i, label in enumerate(label_names):
        label_metrics.append({
            'Label': label,
            'Precision': precision_score(y_test[:, i], y_pred.toarray()[:, i], zero_division=0),
            'Recall': recall_score(y_test[:, i], y_pred.toarray()[:, i], zero_division=0),
            'F1': f1_score(y_test[:, i], y_pred.toarray()[:, i], zero_division=0)
        })

    label_df = pd.DataFrame(label_metrics)
    plt.figure(figsize=(10, 6))
    sns.heatmap(label_df.set_index('Label'), annot=True, cmap='YlGnBu', fmt=".2f")
    plt.title("各慢性病标签预测性能")
    plt.tight_layout()
    plt.savefig('mlknn_label_performance22.png', dpi=300)
    plt.close()

    return best_model, best_params, results_df


# ---------------------------
# 4. 执行代码
# ---------------------------
if __name__ == "__main__":
    # 文件路径配置
    data_path = "训练集备份.xlsx"

    # 运行模型
    best_model, best_params, results_df = mlknn_modeling(data_path)

    print("\n建模完成！")
    print(f"最佳参数: {best_params}")
    print(f"详细结果已保存到 mlknn_parameter_results.xlsx")
    print(f"各标签性能热力图已保存为 mlknn_label_performance.png")