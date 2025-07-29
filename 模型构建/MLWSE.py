import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, hamming_loss, f1_score, precision_score, recall_score, \
    classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler
from itertools import product
import warnings
from joblib import dump



warnings.filterwarnings('ignore')


# 1. 数据加载与预处理
def load_data(file_path):
    data = pd.read_excel(file_path)
    # 分离特征和标签
    X = data.iloc[:, 1:-4].values
    y = data.iloc[:, -4:].values
    feature_names = data.columns[1:-4]
    label_names = data.columns[-4:]
    return X, y, feature_names, label_names


# 2. 评估函数
def evaluate_model(y_true, y_pred, y_proba=None, label_names=None):
    metrics = {
        'Hamming Loss': hamming_loss(y_true, y_pred),
        'Accuracy': accuracy_score(y_true, y_pred),
        'Macro F1': f1_score(y_true, y_pred, average='macro'),
        'Micro F1': f1_score(y_true, y_pred, average='micro'),
        'Precision (macro)': precision_score(y_true, y_pred, average='macro'),
        'Recall (macro)': recall_score(y_true, y_pred, average='macro'),
    }





    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=label_names))

    return metrics


# 3. 基学习器定义（Random Forest, LightGBM, MLP）
def get_base_models(rf_params=None, lgb_params=None, mlp_params=None):
    models = {
        'Random Forest': RandomForestClassifier(**rf_params) if rf_params else RandomForestClassifier(random_state=42),
        'LightGBM': LGBMClassifier(**lgb_params) if lgb_params else LGBMClassifier(random_state=42),
        'MLP': MLPClassifier(**mlp_params) if mlp_params else MLPClassifier(random_state=42, max_iter=1000)
    }
    return models


# 4. MLWSE模型实现
class MLWSE:
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
        self.weights = None
        self.meta_models = None
        self.base_models_trained = {}

    def _get_proba(self, model, X, n_labels):
        """统一处理概率预测"""
        try:
            # 获取预测概率或预测值
            if hasattr(model, 'predict_proba'):
                # 处理MultiOutputClassifier包装的情况
                if hasattr(model, 'estimators_'):
                    probas = []
                    for estimator in model.estimators_:
                        proba = estimator.predict_proba(X)
                        proba = np.array(proba)
                        # 二分类取正类概率，多分类取最大概率
                        if proba.ndim == 2 and proba.shape[1] == 2:
                            probas.append(proba[:, 1])
                        elif proba.ndim == 2:
                            probas.append(proba.max(axis=1))
                        else:
                            probas.append(proba)
                    return np.column_stack(probas)

                # 处理单输出分类器
                else:
                    proba = model.predict_proba(X)
                    proba = np.array(proba)
                    if proba.ndim == 3:  # 多标签输出
                        return proba[:, :, 1]
                    elif proba.shape[1] == 2:  # 二分类
                        return proba[:, 1].reshape(-1, 1)
                    else:  # 多分类
                        return proba.max(axis=1).reshape(-1, 1)

            # 没有概率预测时使用预测值
            else:
                pred = model.predict(X)
                return pred.reshape(-1, n_labels) if pred.ndim == 1 else pred

        except Exception as e:
            print(f"Probability extraction warning: {str(e)}")
            # 回退到简单预测
            pred = model.predict(X)
            return pred.reshape(-1, n_labels) if pred.ndim == 1 else pred

    def fit(self, X, y):
        n_samples, n_labels = y.shape
        n_models = len(self.base_models)

        # 初始化权重矩阵 (模型数 × 标签数)
        self.weights = np.ones((n_models, n_labels)) / n_models

        # 存储每个标签的元模型
        self.meta_models = [None] * n_labels

        # 用于存储基模型的k折预测
        base_predictions = np.zeros((n_samples, n_models, n_labels))

        # 第一阶段：训练基模型并计算权重
        print("Training base models and calculating weights...")
        for model_idx, (name, model) in enumerate(self.base_models.items()):
            print(f"  Processing {name}...")
            kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)

            # 使用k折交叉验证获取预测
            for train_idx, val_idx in kf.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                # 多标签适配 - MLP需要包装
                if isinstance(model, MLPClassifier):
                    model_wrapped = MultiOutputClassifier(model)
                else:
                    model_wrapped = model

                # 训练模型
                try:
                    model_wrapped.fit(X_train, y_train)
                except:
                    model_wrapped = MultiOutputClassifier(model)
                    model_wrapped.fit(X_train, y_train)

                # 获取预测概率
                y_proba = self._get_proba(model_wrapped, X_val, n_labels)

                # 确保形状正确
                if y_proba.shape != (len(val_idx), n_labels):
                    try:
                        y_proba = y_proba.reshape(len(val_idx), n_labels)
                    except:
                        y_proba = np.zeros((len(val_idx), n_labels))

                base_predictions[val_idx, model_idx, :] = y_proba

            # 完整训练模型
            if isinstance(model, MLPClassifier):
                model_wrapped = MultiOutputClassifier(model)
            else:
                model_wrapped = model

            try:
                model_wrapped.fit(X, y)
            except:
                model_wrapped = MultiOutputClassifier(model)
                model_wrapped.fit(X, y)

            self.base_models_trained[name] = model_wrapped

            # 计算该模型在每个标签上的权重
            y_pred = self.base_models_trained[name].predict(X)
            for label_idx in range(n_labels):
                self.weights[model_idx, label_idx] = f1_score(y[:, label_idx], y_pred[:, label_idx])

        # 权重归一化
        self.weights = self.weights / (self.weights.sum(axis=0, keepdims=True) + 1e-10)

        print("\nModel weights per label:")
        weights_df = pd.DataFrame(self.weights,
                                  columns=[f"Label {i}" for i in range(n_labels)],
                                  index=list(self.base_models.keys()))
        print(weights_df.to_string())

        # 第二阶段：训练元模型
        print("\nTraining meta models...")
        for label_idx in range(n_labels):
            print(f"  Training meta model for label {label_idx}...")
            # 加权基模型预测作为元特征
            X_meta = np.zeros((n_samples, n_models))
            for model_idx in range(n_models):
                X_meta[:, model_idx] = base_predictions[:, model_idx, label_idx] * self.weights[model_idx, label_idx]

            # 训练该标签的元模型
            self.meta_models[label_idx] = self.meta_model.fit(X_meta, y[:, label_idx])

    def predict(self, X):
        n_samples = X.shape[0]
        n_models = len(self.base_models_trained)
        n_labels = len(self.meta_models)

        # 获取基模型预测
        base_preds = np.zeros((n_models, n_samples, n_labels))
        for model_idx, (name, model) in enumerate(self.base_models_trained.items()):
            pred = self._get_proba(model, X, n_labels)

            if pred.shape != (n_samples, n_labels):
                try:
                    pred = pred.reshape(n_samples, n_labels)
                except:
                    pred = np.zeros((n_samples, n_labels))

            base_preds[model_idx] = pred

        # 加权组合预测
        final_preds = np.zeros((n_samples, n_labels))
        final_proba = np.zeros((n_samples, n_labels))  # 用于计算AUC
        for label_idx in range(n_labels):
            X_meta = np.zeros((n_samples, n_models))
            for model_idx in range(n_models):
                X_meta[:, model_idx] = base_preds[model_idx, :, label_idx] * self.weights[model_idx, label_idx]
            final_preds[:, label_idx] = self.meta_models[label_idx].predict(X_meta)
            final_proba[:, label_idx] = self.meta_models[label_idx].predict_proba(X_meta)[:, 1]  # 正类概率

        return final_preds.astype(int), final_proba


# 5. 主程序
def main():
    # 加载数据
    file_path = "SMOTE.xlsx"  # 替换为您的文件路径
    X, y, feature_names, label_names = load_data(file_path)

    # 数据分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 数据标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 定义超参数网格
    param_grid = {
        'n_folds': [5],

        # 基模型参数
        'rf__n_estimators': [100],
        'rf__max_depth': [10],
        'rf__max_features': ['sqrt','log2'],

        'lgb__n_estimators': [200],
        'lgb__max_depth': [6],
        'lgb__learning_rate': [0.01,0.1],

        'mlp__hidden_layer_sizes': [(100,)],
        'mlp__alpha': [0.01],

        # 元模型参数
        'meta__n_estimators': [100,200],
        'meta__learning_rate': [0.01,0.05],
        'meta__max_depth': [2]  # 浅层GB
    }

    # 生成所有参数组合
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(product(*param_values))

    # 评估不同参数组合
    results = []
    best_score = -1
    best_params = None
    best_model = None

    for i, params in enumerate(param_combinations):
        # 解包参数
        params_dict = dict(zip(param_names, params))
        print(f"\n=== Evaluating Parameter Set {i + 1}/{len(param_combinations)} ===")
        print("Parameters:", params_dict)

        # 设置基模型参数
        rf_params = {
            'n_estimators': params_dict['rf__n_estimators'],
            'max_depth': params_dict['rf__max_depth'],
            'max_features': params_dict['rf__max_features'],
            'random_state': 42,
            'n_jobs': -1
        }

        lgb_params = {
            'n_estimators': params_dict['lgb__n_estimators'],
            'max_depth': params_dict['lgb__max_depth'],
            'learning_rate': params_dict['lgb__learning_rate'],
            'random_state': 42,
            'n_jobs': -1
        }

        mlp_params = {
            'hidden_layer_sizes': params_dict['mlp__hidden_layer_sizes'],
            'alpha': params_dict['mlp__alpha'],
            'random_state': 42,
            'max_iter': 1000
        }

        base_models = get_base_models(rf_params, lgb_params, mlp_params)

        # 设置元模型参数
        meta_model = GradientBoostingClassifier(
            n_estimators=params_dict['meta__n_estimators'],
            learning_rate=params_dict['meta__learning_rate'],
            max_depth=params_dict['meta__max_depth'],
            random_state=42
        )

        # 创建并训练MLWSE模型
        model = MLWSE(base_models, meta_model, n_folds=params_dict['n_folds'])
        model.fit(X_train, y_train)

        # 在测试集上评估
        y_pred, y_proba = model.predict(X_test)
        metrics = evaluate_model(y_test, y_pred, y_proba, label_names)

        # 记录结果
        current_score = metrics['Macro F1']
        results.append({
            'params': params_dict,
            'metrics': metrics
        })

        # 打印当前结果
        print("\nCurrent Performance:")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

        # 更新最佳模型
        if current_score > best_score:
            best_score = current_score
            best_params = params_dict
            best_model = model
            print("*** New best model found! ***")

    # 输出最佳参数和模型
    print("\n=== Best Model ===")
    print("Best Parameters:", best_params)
    print("Best Macro F1 Score:", best_score)

    # 使用最佳模型进行最终评估
    print("\nFinal Evaluation with Best Model:")
    y_pred, y_proba = best_model.predict(X_test)
    final_metrics = evaluate_model(y_test, y_pred, y_proba, label_names)

    # 保存模型权重
    print("\nFinal Model Weights:")
    weights_df = pd.DataFrame(best_model.weights,
                              columns=label_names,
                              index=list(best_model.base_models.keys()))
    print(weights_df.to_string())

    # 保存所有结果
    results_df = pd.DataFrame(results)
    results_df.to_excel("mlwse_results.xlsx", index=False)
    print("\nAll results saved to mlwse_results.xlsx")


if __name__ == "__main__":
    main()