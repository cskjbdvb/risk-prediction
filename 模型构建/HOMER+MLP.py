import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, ParameterGrid, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    f1_score,
    hamming_loss,
    recall_score,
    roc_auc_score
)
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import warnings
import time
import os
import tempfile

warnings.filterwarnings('ignore')
np.random.seed(42)


# 检查文件是否被占用的辅助函数
def is_file_locked(file_path):
    """检查文件是否被占用"""
    try:
        with tempfile.NamedTemporaryFile(dir=os.path.dirname(file_path), delete=True):
            pass
        return False
    except (PermissionError, OSError):
        return True


# 安全保存Excel文件的辅助函数
def safe_save_excel(df, file_path, max_attempts=3):
    """安全保存Excel文件，处理文件被占用的情况"""
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


# HOMER模型实现，继承BaseEstimator和ClassifierMixin
class HOMER(BaseEstimator, ClassifierMixin):
    """HOMER (Hierarchical Output Method based on Ranking) 多标签分类器"""

    def __init__(self, base_estimator=None, order=None, random_state=None):
        se_estimator = base_estimator
        self.order = order
        self.random_state = random_state

    def fit(self, X, y):
        """训练HOMER模型"""
        # 验证输入
        X, y = check_X_y(X, y, multi_output=True)

        n_samples, n_labels = y.shape

        # 确定标签处理顺序
        if self.order is None:
            if self.random_state is not None:
                np.random.seed(self.random_state)
            self.label_order = np.random.permutation(n_labels)
        else:
            self.label_order = self.order

        # 为每个标签训练一个分类器
        self.models_ = []
        X_transformed = X.copy()

        for label_idx in self.label_order:
            # 复制基础分类器
            model = clone(self.base_estimator)

            # 训练当前标签的分类器
            model.fit(X_transformed, y[:, label_idx])
            self.models_.append(model)

            # 将当前标签的预测结果添加到特征中
            if len(self.models_) < n_labels:
                pred = model.predict_proba(X_transformed)[:, 1].reshape(-1, 1)
                X_transformed = np.hstack((X_transformed, pred))

        self.n_features_in_ = X.shape[1]
        self.n_labels_ = n_labels
        return self

    def predict(self, X):
        """预测多标签"""
        # 验证输入
        check_is_fitted(self)
        X = check_array(X)

        n_samples = X.shape[0]
        y_pred = np.zeros((n_samples, self.n_labels_))
        X_transformed = X.copy()

        for i, label_idx in enumerate(self.label_order):
            model = self.models_[i]
            pred = model.predict(X_transformed)
            y_pred[:, label_idx] = pred

            # 将当前标签的预测结果添加到特征中
            if i < self.n_labels_ - 1:
                pred_proba = model.predict_proba(X_transformed)[:, 1].reshape(-1, 1)
                X_transformed = np.hstack((X_transformed, pred_proba))

        return y_pred

    def predict_proba(self, X):
        """预测多标签的概率"""
        # 验证输入
        check_is_fitted(self)
        X = check_array(X)

        n_samples = X.shape[0]
        y_prob = np.zeros((n_samples, self.n_labels_))
        X_transformed = X.copy()

        for i, label_idx in enumerate(self.label_order):
            model = self.models_[i]
            prob = model.predict_proba(X_transformed)[:, 1]
            y_prob[:, label_idx] = prob

            # 将当前标签的预测结果添加到特征中
            if i < self.n_labels_ - 1:
                prob = prob.reshape(-1, 1)
                X_transformed = np.hstack((X_transformed, prob))

        return y_prob

    def get_params(self, deep=True):

        return {
            'base_estimator': self.base_estimator,
            'order': self.order,
            'random_state': self.random_state
        }

    def set_params(self, **params):

        for parameter, value in params.items():
            setattr(self, parameter, value)
        return self


# 导入clone函数
from sklearn.base import clone


# 1. 数据加载和预处理
def load_data(file_path):
    """加载Excel数据并划分特征和标签"""
    try:
        # 读取Excel文件
        data = pd.read_excel(file_path, engine='openpyxl')

        X = data.iloc[:, 1:-4].values  # 特征
        y = data.iloc[:, -4:].values  # 多标签

        # 检查数据是否包含缺失值
        if np.isnan(X).any() or np.isnan(y).any():
            raise ValueError("数据中包含缺失值，请先处理缺失值")

        return X, y
    except Exception as e:
        print(f"加载数据时出错: {str(e)}")
        print("请检查:")
        print("1. 文件路径是否正确")
        print("2. 文件是否为Excel格式(.xlsx)")
        print("3. 数据格式是否正确(第一列ID，中间特征，最后四列标签)")
        print("4. 数据是否完整无缺失值")
        raise


# 2. 数据标准化
def standardize_data(X_train, X_test):
    """标准化特征数据"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


# 3. 多标签评估指标计算
def evaluate_multilabel(y_true, y_pred, y_prob=None):
    """计算多标签分类的各种评估指标"""
    metrics = {}

    # 基本指标
    metrics['Accuracy'] = accuracy_score(y_true, y_pred)
    metrics['Macro Precision'] = precision_score(y_true, y_pred, average='macro')
    metrics['Micro-F1'] = f1_score(y_true, y_pred, average='micro')
    metrics['Macro-F1'] = f1_score(y_true, y_pred, average='macro')
    metrics['Hamming Loss'] = hamming_loss(y_true, y_pred)
    metrics['Macro Recall'] = recall_score(y_true, y_pred, average='macro')

    # ROC AUC (macro)
    if y_prob is not None:
        metrics['ROC AUC (Macro)'] = roc_auc_score(y_true, y_prob, average='macro')

    return metrics


# 4. 五折交叉验证评估（返回每一折的详细指标）
def cross_validation_evaluation(model, X, y):
    """使用五折交叉验证评估模型并返回每一折的详细指标"""
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    all_fold_metrics = []

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
        print(f"\nProcessing Fold {fold_idx + 1}/5...")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # 标准化数据
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 克隆模型并训练
        model_clone = clone(model)
        start_time = time.time()
        model_clone.fit(X_train_scaled, y_train)
        train_time = time.time() - start_time

        # 预测
        y_pred = model_clone.predict(X_test_scaled)

        # 获取预测概率
        y_prob = model_clone.predict_proba(X_test_scaled)

        # 评估
        fold_metrics = evaluate_multilabel(y_test, y_pred, y_prob)
        fold_metrics['Fold'] = fold_idx + 1
        fold_metrics['Train Time (s)'] = train_time

        all_fold_metrics.append(fold_metrics)

        # 打印当前折的指标
        print(f"Fold {fold_idx + 1} Metrics:")
        for metric, value in fold_metrics.items():
            if metric != 'Fold':
                print(f"  {metric}: {value:.4f}")

    return all_fold_metrics


# 5. 保存结果到Excel
def save_results_to_excel(all_fold_metrics, best_params, filename='HOMER_MLP_results.xlsx'):
    """将结果保存到Excel文件"""
    # 创建结果DataFrame
    results_df = pd.DataFrame(all_fold_metrics)

    # 保存所有折的结果
    safe_save_excel(results_df, filename)

    # 提取每个指标的五个值（用于箱线图）
    metrics_to_export = [
        'Accuracy', 'Macro Precision', 'Micro-F1', 'Macro-F1',
        'Hamming Loss', 'Macro Recall', 'ROC AUC (Macro)'
    ]

    metrics_data = {metric: results_df[metric].tolist() for metric in metrics_to_export}
    metrics_df = pd.DataFrame(metrics_data)

    # 提取文件名的基本部分（不含扩展名）
    base_filename = os.path.splitext(filename)[0]
    boxplot_filename = f"{base_filename}_metrics_for_boxplot.xlsx"

    # 保存用于箱线图的数据
    safe_save_excel(metrics_df, boxplot_filename)

    print(f"\n所有折的详细结果已保存到 {filename}")
    print(f"每个指标的五个值已保存到 {boxplot_filename}，可用于构建箱线图")


# 6. 主函数
def main():
    # 加载数据
    file_path = "备份.xlsx"

    try:
        X, y = load_data(file_path)
    except Exception as e:
        print(f"无法加载数据文件: {str(e)}")
        return

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # 数据标准化
    X_train_scaled, X_test_scaled, scaler = standardize_data(X_train, X_test)

    # 定义基础分类器 (MLP)
    base_mlp = MLPClassifier(random_state=42, early_stopping=True)

    # 定义HOMER模型
    homer = HOMER(base_mlp, random_state=42)

    # 定义MLP超参数网格
    param_grid = {
        'base_estimator__hidden_layer_sizes': [(100,)],
        'base_estimator__activation': ['relu'],
        'base_estimator__alpha': [0.0001],
        'base_estimator__learning_rate_init': [0.01],
        'base_estimator__batch_size': [64]
    }

    print("开始HOMER+MLP网格搜索...")

    best_score = -1
    best_params = None
    best_model = None

    # 手动实现网格搜索
    for params in ParameterGrid(param_grid):
        print(f"\nTesting parameters: {params}")
        model = HOMER(
            MLPClassifier(
                random_state=42,
                early_stopping=True,
                **{k.replace('base_estimator__', ''): v for k, v in params.items()}
            ),
            random_state=42
        )

        # 在训练集上进行交叉验证
        fold_metrics = cross_validation_evaluation(model, X_train_scaled, y_train)

        # 计算平均Macro-F1作为评分标准
        avg_macro_f1 = np.mean([fold['Macro-F1'] for fold in fold_metrics])
        print(f"Average Macro-F1: {avg_macro_f1:.4f}")

        if avg_macro_f1 > best_score:
            best_score = avg_macro_f1
            best_params = params
            best_model = model

    print("\n最佳参数:")
    for param, value in best_params.items():
        print(f"{param}: {value}")

    # 使用最佳参数的模型在测试集上评估
    best_model = HOMER(
        MLPClassifier(
            random_state=42,
            early_stopping=True,
            **{k.replace('base_estimator__', ''): v for k, v in best_params.items()}
        ),
        random_state=42
    )

    best_model.fit(X_train_scaled, y_train)
    y_pred = best_model.predict(X_test_scaled)

    # 获取预测概率
    y_prob = best_model.predict_proba(X_test_scaled)

    # 评估测试集
    test_metrics = evaluate_multilabel(y_test, y_pred, y_prob)

    print("\n测试集评估结果:")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")

    # 在整个训练集上进行交叉验证以获取所有折的指标
    print("\n开始五折交叉验证评估...")
    all_fold_metrics = cross_validation_evaluation(best_model, X_train_scaled, y_train)

    # 保存结果
    save_results_to_excel(all_fold_metrics, best_params)


if __name__ == "__main__":
    main()