import matplotlib
matplotlib.use('Agg')  # 在导入其他库前设置非交互式后端，避免PyCharm兼容性问题
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import shap
import seaborn as sns
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
from joblib import dump, load
import os

# ========== 中文字体设置 ==========
# 替换为你系统存在的字体文件路径（示例为Windows路径）
font_path = 'C:\\Windows\\Fonts\\msyhl.ttc'  # 微软雅黑字体
# font_path = 'C:/Windows/Fonts/simhei.ttf'  # 黑体

# 注册字体到Matplotlib
font_manager.fontManager.addfont(font_path)
plt.rcParams['font.family'] = font_manager.FontProperties(fname=font_path).get_name()
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# ========== Seaborn配置 ==========
sns.set_theme(
    style="whitegrid",
    font=plt.rcParams['font.family'],  # 继承Matplotlib的字体设置
    rc={'axes.unicode_minus': False}  # 确保Seaborn不覆盖负号设置
)

warnings.filterwarnings('ignore')

# 1. 数据加载与预处理
def load_data(file_path):
    data = pd.read_excel(file_path)
    X = data.iloc[:, 1:-4].values
    y = data.iloc[:, -4:].values
    feature_names = data.columns[1:-4].tolist()
    label_names = data.columns[-4:].tolist()
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

    if y_proba is not None:
        try:
            metrics['ROC AUC (macro)'] = roc_auc_score(y_true, y_proba, average='macro')
        except:
            pass

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=label_names))
    return metrics


# 3. 基学习器定义
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
        try:
            if hasattr(model, 'predict_proba'):
                if hasattr(model, 'estimators_'):
                    probas = []
                    for estimator in model.estimators_:
                        proba = estimator.predict_proba(X)
                        proba = np.array(proba)
                        if proba.ndim == 2 and proba.shape[1] == 2:
                            probas.append(proba[:, 1])
                        elif proba.ndim == 2:
                            probas.append(proba.max(axis=1))
                        else:
                            probas.append(proba)
                    return np.column_stack(probas)
                else:
                    proba = model.predict_proba(X)
                    proba = np.array(proba)
                    if proba.ndim == 3:
                        return proba[:, :, 1]
                    elif proba.shape[1] == 2:
                        return proba[:, 1].reshape(-1, 1)
                    else:
                        return proba.max(axis=1).reshape(-1, 1)
            else:
                pred = model.predict(X)
                return pred.reshape(-1, n_labels) if pred.ndim == 1 else pred
        except Exception as e:
            print(f"Probability extraction warning: {str(e)}")
            pred = model.predict(X)
            return pred.reshape(-1, n_labels) if pred.ndim == 1 else pred

    def fit(self, X, y):
        n_samples, n_labels = y.shape
        n_models = len(self.base_models)
        self.weights = np.ones((n_models, n_labels)) / n_models
        self.meta_models = [None] * n_labels
        base_predictions = np.zeros((n_samples, n_models, n_labels))

        print("Training base models and calculating weights...")
        for model_idx, (name, model) in enumerate(self.base_models.items()):
            print(f"  Processing {name}...")
            kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)

            for train_idx, val_idx in kf.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                if isinstance(model, MLPClassifier):
                    model_wrapped = MultiOutputClassifier(model)
                else:
                    model_wrapped = model

                try:
                    model_wrapped.fit(X_train, y_train)
                except:
                    model_wrapped = MultiOutputClassifier(model)
                    model_wrapped.fit(X_train, y_train)

                y_proba = self._get_proba(model_wrapped, X_val, n_labels)
                if y_proba.shape != (len(val_idx), n_labels):
                    try:
                        y_proba = y_proba.reshape(len(val_idx), n_labels)
                    except:
                        y_proba = np.zeros((len(val_idx), n_labels))

                base_predictions[val_idx, model_idx, :] = y_proba

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
            y_pred = self.base_models_trained[name].predict(X)
            for label_idx in range(n_labels):
                self.weights[model_idx, label_idx] = f1_score(y[:, label_idx], y_pred[:, label_idx])

        self.weights = self.weights / (self.weights.sum(axis=0, keepdims=True) + 1e-10)

        print("\nModel weights per label:")
        weights_df = pd.DataFrame(self.weights,
                                  columns=[f"Label {i}" for i in range(n_labels)],
                                  index=list(self.base_models.keys()))
        print(weights_df.to_string())

        print("\nTraining meta models...")
        for label_idx in range(n_labels):
            print(f"  Training meta model for label {label_idx}...")
            X_meta = np.zeros((n_samples, n_models))
            for model_idx in range(n_models):
                X_meta[:, model_idx] = base_predictions[:, model_idx, label_idx] * self.weights[model_idx, label_idx]
            self.meta_models[label_idx] = self.meta_model.fit(X_meta, y[:, label_idx])

    def predict(self, X):
        n_samples = X.shape[0]
        n_models = len(self.base_models_trained)
        n_labels = len(self.meta_models)

        base_preds = np.zeros((n_models, n_samples, n_labels))
        for model_idx, (name, model) in enumerate(self.base_models_trained.items()):
            pred = self._get_proba(model, X, n_labels)
            if pred.shape != (n_samples, n_labels):
                try:
                    pred = pred.reshape(n_samples, n_labels)
                except:
                    pred = np.zeros((n_samples, n_labels))
            base_preds[model_idx] = pred

        final_preds = np.zeros((n_samples, n_labels))
        final_proba = np.zeros((n_samples, n_labels))
        for label_idx in range(n_labels):
            X_meta = np.zeros((n_samples, n_models))
            for model_idx in range(n_models):
                X_meta[:, model_idx] = base_preds[model_idx, :, label_idx] * self.weights[model_idx, label_idx]
            final_preds[:, label_idx] = self.meta_models[label_idx].predict(X_meta)
            try:
                final_proba[:, label_idx] = self.meta_models[label_idx].predict_proba(X_meta)[:, 1]
            except:
                final_proba[:, label_idx] = final_preds[:, label_idx]

        return final_preds.astype(int), final_proba

    def plot_shap_summary(self, X, feature_names, label_names, save_dir="SHAP_Results"):
        """绘制SHAP特征重要性摘要图"""
        os.makedirs(save_dir, exist_ok=True)


        # 对每个标签进行分析
        for label_idx, label_name in enumerate(label_names):
            print(f"\nAnalyzing SHAP for {label_name}...")

            # 定义预测函数
            def predict_fn(X):
                n_samples = X.shape[0]
                n_models = len(self.base_models_trained)
                base_preds = np.zeros((n_models, n_samples))

                for model_idx, (name, model) in enumerate(self.base_models_trained.items()):
                    pred = self._get_proba(model, X, len(label_names))
                    if pred.shape != (n_samples, len(label_names)):
                        try:
                            pred = pred.reshape(n_samples, len(label_names))
                        except:
                            pred = np.zeros((n_samples, len(label_names)))
                    base_preds[model_idx] = pred[:, label_idx] * self.weights[model_idx, label_idx]

                # 加权平均预测
                return base_preds.mean(axis=0)

            # 计算SHAP值
            explainer = shap.KernelExplainer(predict_fn, shap.sample(X, 50))
            shap_values = explainer.shap_values(X[:100])  # 限制样本数量以加快计算

            # 确保shap_values是二维数组
            shap_values = np.array(shap_values)
            if len(shap_values.shape) == 1:
                shap_values = shap_values.reshape(1, -1)

            # 计算平均绝对SHAP值
            mean_shap = np.abs(shap_values).mean(axis=0)

            # 创建DataFrame并排序
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': mean_shap
            }).sort_values('Importance', ascending=True)

            # 绘制横向条形图
            plt.figure(figsize=(12, 0.5 * len(feature_names)))
            plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
            plt.xlabel('平均绝对SHAP值', fontsize=12)
            plt.title(f'特征重要性 - {label_name}', fontsize=14)
            plt.tight_layout()
            plt.savefig(f"{save_dir}/SHAP_Importance_{label_name}.png", dpi=300, bbox_inches='tight')
            plt.close()

        # 绘制综合特征重要性
        self._plot_combined_shap(X, feature_names, label_names, save_dir)

    def _plot_combined_shap(self, X, feature_names, label_names, save_dir):
        """绘制四个标签的综合特征重要性"""
        combined_shap = np.zeros(len(feature_names))

        # 计算每个标签的SHAP值并累加
        for label_idx in range(len(label_names)):
            def predict_fn(X):
                n_samples = X.shape[0]
                n_models = len(self.base_models_trained)
                base_preds = np.zeros((n_models, n_samples))

                for model_idx, (name, model) in enumerate(self.base_models_trained.items()):
                    pred = self._get_proba(model, X, len(label_names))
                    if pred.shape != (n_samples, len(label_names)):
                        try:
                            pred = pred.reshape(n_samples, len(label_names))
                        except:
                            pred = np.zeros((n_samples, len(label_names)))
                    base_preds[model_idx] = pred[:, label_idx] * self.weights[model_idx, label_idx]

                return base_preds.mean(axis=0)

            explainer = shap.KernelExplainer(predict_fn, shap.sample(X, 50))
            shap_values = explainer.shap_values(X[:100])
            shap_values = np.array(shap_values)
            if len(shap_values.shape) == 1:
                shap_values = shap_values.reshape(1, -1)

            combined_shap += np.abs(shap_values).mean(axis=0)

        # 创建DataFrame并排序
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': combined_shap / len(label_names)  # 平均重要性
        }).sort_values('Importance', ascending=True)

        # 绘制横向条形图
        plt.figure(figsize=(12, 0.5 * len(feature_names)))
        plt.barh(importance_df['Feature'], importance_df['Importance'], color='salmon')
        plt.xlabel('平均绝对SHAP值', fontsize=12)
        plt.title('综合特征重要性 (四个标签平均)', fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/SHAP_Importance_Combined.png", dpi=300, bbox_inches='tight')
        plt.close()


# 5. 主程序
def main():
    # 加载数据
    file_path = "D:\\研二2\\论文撰写\\17-23年数据\\特征选择.xlsx"
    X, y, feature_names, label_names = load_data(file_path)

    # 数据分割和标准化
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 定义超参数
    param_grid = {
        'n_folds': [5],
        'rf__n_estimators': [100],
        'rf__max_depth': [10],
        'rf__max_features': ['sqrt'],
        'lgb__n_estimators': [200],
        'lgb__max_depth': [6],
        'lgb__learning_rate': [0.1],
        'mlp__hidden_layer_sizes': [(100,)],
        'mlp__alpha': [0.01],
        'meta__n_estimators': [100],
        'meta__learning_rate': [0.05],
        'meta__max_depth': [2]
    }

    # 模型训练和评估
    best_model = None
    best_score = -1
    param_combinations = list(product(*param_grid.values()))

    for i, params in enumerate(param_combinations):
        params_dict = dict(zip(param_grid.keys(), params))

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
        meta_model = GradientBoostingClassifier(
            n_estimators=params_dict['meta__n_estimators'],
            learning_rate=params_dict['meta__learning_rate'],
            max_depth=params_dict['meta__max_depth'],
            random_state=42
        )

        model = MLWSE(base_models, meta_model, n_folds=params_dict['n_folds'])
        model.fit(X_train, y_train)

        y_pred, y_proba = model.predict(X_test)
        metrics = evaluate_model(y_test, y_pred, y_proba, label_names)
        current_score = metrics['Macro F1']

        if current_score > best_score:
            best_score = current_score
            best_model = model
            print("*** 发现新的最佳模型 ***")

    # 保存模型
    #model_path = "D:\\研一2\\python\\练习位置\\pythonProject\\小论文数据预处理\\建模\\MLWSE\\MLWSE_model.pkl"
    #dump(best_model, model_path)
    #print(f"\n最佳模型已保存到: {model_path}")

    # 绘制SHAP特征重要性图
    print("\n开始绘制SHAP特征重要性图...")
    best_model.plot_shap_summary(
        X_test[:100],  # 使用少量样本加快计算
        feature_names,
        label_names,
        save_dir="SHAP_Importance_Results66"
    )
    print("\nSHAP特征重要性图已保存到 SHAP_Importance_Results66 目录")


if __name__ == "__main__":
    main()