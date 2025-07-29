import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, train_test_split, RepeatedKFold
from sklearn.metrics import (
    hamming_loss, accuracy_score, f1_score,
    precision_score, recall_score, jaccard_score,
    classification_report
)
import os
import warnings

# ========== 中文显示配置 ==========
font_path = 'C:\\Windows\\Fonts\\msyhl.ttc'
font_manager.fontManager.addfont(font_path)
plt.rcParams['font.family'] = font_manager.FontProperties(fname=font_path).get_name()
plt.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings('ignore')


# ========== 数据加载 ==========
def load_data(file_path):
    """加载数据集并返回特征和标签"""
    data = pd.read_excel(file_path)
    feature_names = data.columns[1:-4].tolist()
    label_names = data.columns[-4:].tolist()
    X = data.iloc[:, 1:-4].values
    y = data.iloc[:, -4:].values
    return X, y, feature_names, label_names



class BR_RF(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.models = []
        self.feature_names = None

    def fit(self, X, y):
        """训练每个标签的随机森林"""
        self.models = []
        for i in range(y.shape[1]):
            model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state
            )
            model.fit(X, y[:, i])
            self.models.append(model)
        return self

    def predict(self, X):
        """预测所有标签"""
        preds = np.zeros((X.shape[0], len(self.models)))
        for i, model in enumerate(self.models):
            preds[:, i] = model.predict(X)
        return preds.astype(int)

    def predict_proba(self, X):
        """获取概率预测"""
        probas = np.zeros((X.shape[0], len(self.models)))
        for i, model in enumerate(self.models):
            probas[:, i] = model.predict_proba(X)[:, 1]
        return probas

    # ========== 群体SHAP分析 ==========
    def plot_shap_analysis(self, X, feature_names, label_names, sample_size=100, save_dir="SHAP_Results"):
        """群体特征重要性分析"""
        os.makedirs(save_dir, exist_ok=True)
        X_sample = X[:sample_size] if X.shape[0] > sample_size else X

        # 各标签独立分析
        for idx, (model, label) in enumerate(zip(self.models, label_names)):
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_sample)
                if isinstance(shap_values, list) and len(shap_values) == 2:
                    shap_values = shap_values[1]

                plt.figure(figsize=(10, 8))
                shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
                plt.title(f"SHAP特征重要性 - {label}", fontsize=14)
                plt.tight_layout()
                plt.savefig(f"{save_dir}/SHAP_{label}.png", dpi=300, bbox_inches='tight')
                plt.close()
            except Exception as e:
                print(f"生成{label}的SHAP图时出错: {str(e)}")

        # 综合特征重要性
        self._plot_combined_importance(X_sample, feature_names, label_names, save_dir)

    def _plot_combined_importance(self, X, feature_names, label_names, save_dir):
        """综合特征重要性可视化"""
        total_importance = np.zeros(len(feature_names))

        for model in self.models:
            try:
                importance = model.feature_importances_
            except:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X)
                if isinstance(shap_values, list) and len(shap_values) == 2:
                    shap_values = shap_values[1]
                importance = np.abs(shap_values).mean(axis=0)
            total_importance += importance

        importance_df = pd.DataFrame({
            "特征": feature_names,
            "重要性": total_importance
        }).sort_values("重要性", ascending=True)

        plt.figure(figsize=(12, 0.5 * len(feature_names)))
        plt.barh(importance_df["特征"], importance_df["重要性"], color="teal")
        plt.xlabel("特征重要性总和", fontsize=12)
        plt.title("综合特征重要性（四标签总和）", fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/综合特征重要性.png", dpi=300, bbox_inches='tight')
        plt.close()

    # ========== 个体SHAP分析 ==========
    # ========== 个体SHAP分析 ==========
    def plot_individual_shap(self, X, sample_index=None, feature_names=None,
                             label_names=None, save_dir="Individual_SHAP"):
        os.makedirs(save_dir, exist_ok=True)
        if sample_index is None:
            sample_index = np.random.randint(0, X.shape[0])
        X_sample = X[sample_index:sample_index + 1]
        print(f"\n分析样本索引: {sample_index}")

        X_background = shap.sample(X, 100) if X.shape[0] > 100 else X

        plt.figure(figsize=(15, 10))
        for idx, (model, label) in enumerate(zip(self.models, label_names)):
            try:
                explainer = shap.TreeExplainer(model, X_background)
                shap_values = explainer.shap_values(X_sample)

                # ==== 修改点：适配SHAP v0.20+的API ====
                if isinstance(shap_values, list):
                    # 二分类模型：取正类（索引1）的SHAP值
                    base_value = explainer.expected_value[1]
                    shap_values = shap_values[1]
                else:
                    base_value = explainer.expected_value

                plt.subplot(2, 2, idx + 1)
                shap.plots.force(
                    base_value,
                    shap_values[0],
                    feature_names=feature_names,
                    matplotlib=True,
                    show=False
                )
                plt.title(f"{label} SHAP贡献", fontsize=12)
            except Exception as e:
                print(f"生成{label}的SHAP图时出错: {str(e)}")

        plt.suptitle(f"样本{sample_index}多标签SHAP解释", fontsize=14, y=1.02)
        plt.savefig(f"{save_dir}/样本{sample_index}_多标签SHAP.png", dpi=300, bbox_inches='tight')
        plt.close()

        pd.DataFrame(X_sample, columns=feature_names).to_csv(
            f"{save_dir}/样本{sample_index}_特征值.csv", index=False)


# ========== 模型评估 ==========
def evaluate_model(model, X, y, label_names):
    """交叉验证评估"""
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    metrics = {
        'Hamming Loss': [], 'Accuracy': [], 'Macro F1': [],
        'Micro F1': [], 'Precision (Macro)': [],
        'Recall (Macro)': [], 'Jaccard Score': []
    }

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        cloned_model = clone(model).fit(X_train, y_train)
        y_pred = cloned_model.predict(X_test)

        metrics['Hamming Loss'].append(hamming_loss(y_test, y_pred))
        metrics['Accuracy'].append(accuracy_score(y_test, y_pred))
        metrics['Macro F1'].append(f1_score(y_test, y_pred, average='macro'))
        metrics['Micro F1'].append(f1_score(y_test, y_pred, average='micro'))
        metrics['Precision (Macro)'].append(precision_score(y_test, y_pred, average='macro'))
        metrics['Recall (Macro)'].append(recall_score(y_test, y_pred, average='macro'))
        metrics['Jaccard Score'].append(jaccard_score(y_test, y_pred, average='samples'))

    return {k: np.mean(v) for k, v in metrics.items()}


# ========== 主程序 ==========
def main():
    # 数据加载
    data_path = "标准化.xlsx"
    X, y, feature_names, label_names = load_data(data_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 参数调优
    best_score, best_params = -1, None
    param_grid = {'n_estimators': [100, 200, 300], 'max_depth': [None, 10, 20]}

    for n_est in param_grid['n_estimators']:
        for max_depth in param_grid['max_depth']:
            model = BR_RF(n_estimators=n_est, max_depth=max_depth, random_state=42)
            metrics = evaluate_model(model, X_train, y_train, label_names)
            if metrics['Macro F1'] > best_score:
                best_score = metrics['Macro F1']
                best_params = {'n_estimators': n_est, 'max_depth': max_depth}

    # 最终模型
    final_model = BR_RF(**best_params, random_state=42).fit(X_train, y_train)
    print("\n测试集分类报告:")
    print(classification_report(y_test, final_model.predict(X_test), target_names=label_names))

    # 可视化分析
    final_model.plot_shap_analysis(X_test[:100], feature_names, label_names)  # 群体分析
    final_model.plot_individual_shap(X_test, feature_names=feature_names,  # 个体分析
                                     label_names=label_names, sample_index=5)  # 分析第5个样本


if __name__ == "__main__":
    main()