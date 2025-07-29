from sklearn.base import clone
from sklearn.model_selection import ParameterGrid, KFold
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import (
    hamming_loss, accuracy_score, f1_score,
    precision_score, recall_score, jaccard_score,
    roc_auc_score
)
from sklearn.neural_network import MLPClassifier
import time
import warnings
import os
import tempfile

warnings.filterwarnings('ignore')


def load_data(file_path):
    data = pd.read_excel(file_path)
    X = data.iloc[:, 1:-4].values  # 特征
    y = data.iloc[:, -4:].values  # 多标签
    return X, y


def is_file_locked(file_path):
    try:
        with tempfile.NamedTemporaryFile(dir=os.path.dirname(file_path), delete=True):
            pass
        return False
    except (PermissionError, OSError):
        return True



def safe_save_excel(df, file_path, max_attempts=3):

    attempt = 0
    while attempt < max_attempts:
        try:
            if is_file_locked(file_path):
                print(f"警告: 文件 {file_path} 被占用，尝试 {attempt + 1}/{max_attempts}...")
                time.sleep(2)
                attempt += 1
                continue

            df.to_excel(file_path, index=False)
            print(f"成功保存文件: {file_path}")
            return True
        except Exception as e:
            print(f"保存文件时出错: {e}")
            attempt += 1
            time.sleep(2)

    print(f"错误: 无法保存文件 {file_path}，已达到最大尝试次数")
    return False


def evaluate_model(model, X, y):
    """评估模型并返回每一折的详细指标"""
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    all_fold_metrics = []

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
        print(f"\nProcessing Fold {fold_idx + 1}/5...")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model_clone = clone(model)
        start_time = time.time()
        model_clone.fit(X_train, y_train)
        train_time = time.time() - start_time

        y_pred = model_clone.predict(X_test)

        # 获取预测概率（用于ROC AUC计算）
        y_prob = model_clone.predict_proba(X_test)

        fold_metrics = {
            'Fold': fold_idx + 1,
            'Hamming Loss': hamming_loss(y_test, y_pred),
            'Accuracy': accuracy_score(y_test, y_pred),
            'Macro F1': f1_score(y_test, y_pred, average='macro'),
            'Micro F1': f1_score(y_test, y_pred, average='micro'),
            'Precision (Macro)': precision_score(y_test, y_pred, average='macro'),
            'Recall (Macro)': recall_score(y_test, y_pred, average='macro'),
            'Jaccard Score': jaccard_score(y_test, y_pred, average='samples'),
            'ROC AUC (Macro)': roc_auc_score(y_test, y_prob, average='macro'),
            'Train Time (s)': train_time
        }

        all_fold_metrics.append(fold_metrics)

        # 打印当前折的指标
        print(f"Fold {fold_idx + 1} Metrics:")
        for metric, value in fold_metrics.items():
            if metric != 'Fold':
                print(f"  {metric}: {value:.4f}")

    return all_fold_metrics


def run_experiment(model_class, param_grid, X, y):
    best_score = -1
    best_params = None
    all_param_results = []

    # 参数网格搜索
    for params in ParameterGrid(param_grid):
        print(f"\nTesting {model_class.__name__} params:", params)
        model = model_class(**params)
        fold_metrics = evaluate_model(model, X, y)

        # 计算平均指标
        avg_metrics = {
            metric: np.mean([fold[metric] for fold in fold_metrics])
            for metric in fold_metrics[0].keys() if metric != 'Fold'
        }

        # 添加参数信息
        for fold in fold_metrics:
            fold.update(params)

        all_param_results.extend(fold_metrics)

        print("\nAverage Metrics:")
        for metric, value in avg_metrics.items():
            print(f"  {metric}: {value:.4f}")

        if avg_metrics['Macro F1'] > best_score:
            best_score = avg_metrics['Macro F1']
            best_params = params

    print("\n=== Best Model ===")
    print("Best Params:", best_params)

    results_df = pd.DataFrame(all_param_results)
    safe_save_excel(results_df, 'br_mlp_cv_results.xlsx')

    best_model_results = results_df.copy()
    for param_name, param_value in best_params.items():
        best_model_results = best_model_results[best_model_results[param_name] == param_value]

    metrics_to_export = [
        'Hamming Loss', 'Accuracy', 'Macro F1', 'Micro F1',
        'Precision (Macro)', 'Recall (Macro)', 'Jaccard Score', 'ROC AUC (Macro)'
    ]

    metrics_data = {metric: best_model_results[metric].tolist() for metric in metrics_to_export}
    metrics_df = pd.DataFrame(metrics_data)
    safe_save_excel(metrics_df, 'br_mlp_metrics_for_boxplot.xlsx')

    print("\nCross-validation metrics for each fold have been saved to 'br_mlp_metrics_for_boxplot.xlsx'")
    print("These values can be used to construct box plots for each metric.")

    # 使用最佳参数训练最终模型
    final_model = model_class(**best_params)
    final_model.fit(X, y)

    return final_model


class BR_MLP(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden_layer_sizes=(100,), activation='relu',
                 solver='adam', alpha=0.0001, random_state=42):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.random_state = random_state
        self.models = []

    def get_params(self, deep=True):
        return {
            'hidden_layer_sizes': self.hidden_layer_sizes,
            'activation': self.activation,
            'solver': self.solver,
            'alpha': self.alpha,
            'random_state': self.random_state
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def fit(self, X, y):
        """训练多个MLP模型，每个模型对应一个标签"""
        self.models = []
        n_labels = y.shape[1]

        for i in range(n_labels):
            print(f"Training MLP for label {i + 1}/{n_labels}")
            model = MLPClassifier(
                hidden_layer_sizes=self.hidden_layer_sizes,
                activation=self.activation,
                solver=self.solver,
                alpha=self.alpha,
                random_state=self.random_state,
                max_iter=300  # 增加最大迭代次数，确保收敛
            )
            # 为每个标签训练一个二分类器
            model.fit(X, y[:, i])
            self.models.append(model)

        return self

    def predict(self, X):
        """预测多个标签"""
        n_samples = X.shape[0]
        n_labels = len(self.models)
        y_pred = np.zeros((n_samples, n_labels))

        for i, model in enumerate(self.models):
            y_pred[:, i] = model.predict(X)

        return y_pred

    def predict_proba(self, X):
        """返回预测概率，用于计算ROC AUC"""
        n_samples = X.shape[0]
        n_labels = len(self.models)
        y_prob = np.zeros((n_samples, n_labels))

        for i, model in enumerate(self.models):
            y_prob[:, i] = model.predict_proba(X)[:, 1]  # 获取正类的概率

        return y_prob


if __name__ == "__main__":
    X, y = load_data("训练集.xlsx")
    param_grid = {
        'hidden_layer_sizes': [(50,), (100,), (50, 50)],
        'activation': ['relu'],
        'solver': ['adam']
    }
    run_experiment(BR_MLP, param_grid, X, y)