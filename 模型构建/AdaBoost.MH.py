import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, ParameterGrid, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    hamming_loss, roc_auc_score
)
from sklearn.pipeline import Pipeline
import time
import warnings

warnings.filterwarnings('ignore')


# 1. 数据加载和预处理
def load_data(file_path, test_size=0.2):
    """加载数据并划分训练集和测试集"""
    data = pd.read_excel(file_path)

    # 分离特征和标签
    X = data.iloc[:, 1:-4].values
    y = data.iloc[:, -4:].values

    # 划分训练集和测试集（按标签和分层抽样）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y.sum(axis=1))

    # 数据标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("\n数据统计信息:")
    print(f"- 训练集样本数: {X_train.shape[0]}, 特征数: {X_train.shape[1]}")
    print(f"- 测试集样本数: {X_test.shape[0]}")
    print("\n标签分布 (训练集):")
    print(pd.DataFrame(y_train, columns=[f'Label_{i}' for i in range(4)]).sum())

    return X_train, X_test, y_train, y_test, scaler


# 2. 评估函数（添加ROC AUC并支持单折评估）
def evaluate_model(y_true, y_pred, y_prob=None):
    """评估多标签模型性能，添加ROC AUC并支持单折评估"""
    metrics = {}

    # 全局指标
    metrics['Accuracy'] = accuracy_score(y_true, y_pred)
    metrics['Macro_Precision'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['Micro_F1'] = f1_score(y_true, y_pred, average='micro', zero_division=0)
    metrics['Macro_F1'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['Hamming_Loss'] = hamming_loss(y_true, y_pred)
    metrics['Macro_Recall'] = recall_score(y_true, y_pred, average='macro', zero_division=0)

    # 添加ROC AUC (macro)评估指标
    if y_prob is not None:
        try:
            metrics['ROC_AUC_Macro'] = roc_auc_score(y_true, y_prob, average='macro')
        except ValueError:
            print("警告: 无法计算ROC AUC，某些标签没有足够的样本。")
            metrics['ROC_AUC_Macro'] = np.nan

    return metrics


# 3. 超参数搜索（修改为使用五折交叉验证）
def parameter_search(X_train, y_train):
    """执行网格参数搜索，使用五折交叉验证"""
    param_grid = {
        'model__estimator__n_estimators': [500],  # AdaBoost参数
        'model__estimator__learning_rate': [1.0],  # AdaBoost参数
        'model__estimator__algorithm': ['SAMME'],  # AdaBoost参数
        'model__estimator__estimator__max_depth': [4]  # 决策树参数
    }

    # 创建基础模型
    base_tree = DecisionTreeClassifier(random_state=42)
    ada = AdaBoostClassifier(
        estimator=base_tree,
        random_state=42
    )
    model = MultiOutputClassifier(ada)

    # 使用管道整合流程
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])

    # 初始化K折交叉验证
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # 存储所有折的结果
    all_fold_results = []
    best_avg_score = -np.inf
    best_params = None
    best_model = None

    print("\n开始参数搜索...")
    for params in ParameterGrid(param_grid):
        print(f"\n测试参数: {params}")

        # 当前参数的所有折的结果
        param_fold_results = []

        # 五折交叉验证
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train.sum(axis=1))):
            print(f"\n正在处理第 {fold_idx + 1}/5 折...")

            # 划分训练集和验证集
            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

            # 设置参数
            pipeline.set_params(**params)

            # 训练模型
            start_time = time.time()
            pipeline.fit(X_fold_train, y_fold_train)
            train_time = time.time() - start_time

            # 预测
            y_fold_pred = pipeline.predict(X_fold_val)

            # 获取预测概率（用于ROC AUC计算）
            y_fold_prob = np.array(
                [est.predict_proba(X_fold_val)[:, 1] for est in pipeline.named_steps['model'].estimators_]).T

            # 评估
            metrics = evaluate_model(y_fold_val, y_fold_pred, y_fold_prob)
            metrics.update({
                'Fold': fold_idx + 1,
                'n_estimators': params['model__estimator__n_estimators'],
                'learning_rate': params['model__estimator__learning_rate'],
                'algorithm': params['model__estimator__algorithm'],
                'max_depth': params['model__estimator__estimator__max_depth'],
                'Time': train_time
            })

            param_fold_results.append(metrics)

            # 打印当前折的结果
            print(f"第 {fold_idx + 1} 折结果:")
            print(
                f"Macro F1: {metrics['Macro_F1']:.4f} | Hamming Loss: {metrics['Hamming_Loss']:.4f} | ROC AUC: {metrics.get('ROC_AUC_Macro', 'N/A')}")
            print(f"训练时间: {metrics['Time']:.2f}s")

        # 计算当前参数的平均得分
        avg_macro_f1 = np.mean([fold['Macro_F1'] for fold in param_fold_results])
        print(f"\n参数 {params} 的平均 Macro F1: {avg_macro_f1:.4f}")

        # 保存所有折的结果
        all_fold_results.extend(param_fold_results)

        # 更新最佳参数
        if avg_macro_f1 > best_avg_score:
            best_avg_score = avg_macro_f1
            best_params = params
            best_model = pipeline
            print("↑ 新的最佳参数")

    # 保存所有折的结果到Excel
    results_df = pd.DataFrame(all_fold_results)
    results_df.to_excel('ada_mh_cv_results11.xlsx', index=False)

    # 使用最佳参数重新训练模型（在整个训练集上）
    print("\n使用最佳参数在整个训练集上训练模型...")
    best_model.set_params(**best_params)
    best_model.fit(X_train, y_train)

    return results_df, best_params, best_model


# 4. 主函数
def main():
    # 文件路径 - 替换为您的实际路径
    data_path = "训练集备份.xlsx"

    # 加载数据
    X_train, X_test, y_train, y_test, scaler = load_data(data_path)

    # 参数搜索（使用五折交叉验证）
    results_df, best_params, model = parameter_search(X_train, y_train)

    # 打印最佳参数组合
    print("\n最佳参数组合:")
    print(f"n_estimators: {best_params['model__estimator__n_estimators']}")
    print(f"learning_rate: {best_params['model__estimator__learning_rate']}")
    print(f"algorithm: {best_params['model__estimator__algorithm']}")
    print(f"max_depth: {best_params['model__estimator__estimator__max_depth']}")

    # 在测试集上评估
    print("\n在测试集上评估...")
    y_pred = model.predict(X_test)

    # 获取预测概率（用于ROC AUC计算）
    y_prob = np.array([est.predict_proba(X_test)[:, 1] for est in model.named_steps['model'].estimators_]).T

    # 评估
    test_metrics = evaluate_model(y_test, y_pred, y_prob)

    print("\n测试集性能:")
    print("{:<20} {:<10}".format('Metric', 'Value'))
    print("-" * 30)
    for k, v in test_metrics.items():
        print("{:<20} {:.4f}".format(k, v))

    # 保存所有折的评估结果
    cv_metrics = ['Accuracy', 'Macro_Precision', 'Micro_F1', 'Macro_F1', 'Hamming_Loss', 'Macro_Recall',
                  'ROC_AUC_Macro']
    cv_results = {}

    for metric in cv_metrics:
        if metric in results_df.columns:
            cv_results[metric] = results_df[metric].tolist()

    # 保存交叉验证结果
    cv_df = pd.DataFrame(cv_results)
    cv_df.to_excel('ada_mh_cv_metrics.xlsx', index=False)

    print("\n交叉验证结果已保存到 'ada_mh_cv_metrics.xlsx'")
    print("每个指标的五个值可用于后续构建箱线图")


if __name__ == '__main__':
    main()