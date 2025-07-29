import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import (
    hamming_loss, accuracy_score, f1_score,
    precision_score, recall_score, jaccard_score,
    roc_auc_score
)
from sklearn.base import clone
import time
import warnings

warnings.filterwarnings('ignore')


def load_data(file_path):
    data = pd.read_excel(file_path)
    X = data.iloc[:, 1:-4].values
    y = data.iloc[:, -4:].values
    return X, y


class BR_RF(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.models = []

    def get_params(self, deep=True):
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'random_state': self.random_state
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def fit(self, X, y):
        n_labels = y.shape[1]
        self.models = []
        for i in range(n_labels):
            rf = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state
            )
            rf.fit(X, y[:, i])
            self.models.append(rf)
        return self

    def predict(self, X):
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
            y_prob[:, i] = model.predict_proba(X)[:, 1]  # 正类的概率
        return y_prob


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


def main():
    X, y = load_data("训练集.xlsx")

    param_grid = {
        'n_estimators': [200, 300],
        'max_depth': [20, 40]
    }

    best_score = -1
    best_params = None
    all_param_results = []

    for n_est in param_grid['n_estimators']:
        for max_depth in param_grid['max_depth']:
            params = {
                'n_estimators': n_est,
                'max_depth': max_depth,
                'random_state': 42
            }
            print(f"\nTesting RF params: {params}")

            model = BR_RF(**params)
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

    # 保存所有折的结果到Excel
    results_df = pd.DataFrame(all_param_results)
    results_df.to_excel('br_rf_cv_results.xlsx', index=False)

    # 提取每个指标的五个值（只使用最佳参数的结果）
    best_model_results = results_df[
        (results_df['n_estimators'] == best_params['n_estimators']) &
        (results_df['max_depth'] == best_params['max_depth'])
        ]

    metrics_to_export = [
        'Hamming Loss', 'Accuracy', 'Macro F1', 'Micro F1',
        'Precision (Macro)', 'Recall (Macro)', 'Jaccard Score', 'ROC AUC (Macro)'
    ]

    metrics_data = {metric: best_model_results[metric].tolist() for metric in metrics_to_export}
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.to_excel('br_rf_metrics_for_boxplot.xlsx', index=False)

    print("\nCross-validation metrics for each fold have been saved to 'br_rf_metrics_for_boxplot.xlsx'")
    print("These values can be used to construct box plots for each metric.")


if __name__ == "__main__":
    main()