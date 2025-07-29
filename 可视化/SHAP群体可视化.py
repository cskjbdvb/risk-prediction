import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import (
    hamming_loss,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    jaccard_score,
    classification_report
)
from sklearn.base import clone
import os
import warnings

# ========== 中文显示配置 ==========
font_path = 'C:\\Windows\\Fonts\\msyhl.ttc'  # 微软雅黑字体示例
font_manager.fontManager.addfont(font_path)
plt.rcParams['font.family'] = font_manager.FontProperties(fname=font_path).get_name()
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# ========== 字体大小配置 ==========
SHAP_TITLE_FONT_SIZE = 20  # SHAP摘要图标题字体大小
SHAP_FEATURE_FONT_SIZE = 18  # SHAP摘要图特征名称字体大小
COMBINED_TITLE_FONT_SIZE = 20  # 综合特征重要性图标题字体大小
COMBINED_AXIS_FONT_SIZE = 16  # 综合特征重要性图坐标轴字体大小
COMBINED_FEATURE_FONT_SIZE = 16  # 综合特征重要性图特征名称字体大小
REPORT_TITLE_FONT_SIZE = 16  # 分类报告标题字体大小

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
        self.feature_names = None  # 存储特征名称用于可视化

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
        """获取概率预测（用于SHAP分析）"""
        probas = np.zeros((X.shape[0], len(self.models)))
        for i, model in enumerate(self.models):
            probas[:, i] = model.predict_proba(X)[:, 1]
        return probas

    def plot_shap_analysis(self, X, feature_names, label_names, sample_size=100, save_dir="SHAP_Results"):
        """执行SHAP分析并保存可视化结果"""
        os.makedirs(save_dir, exist_ok=True)

        # 限制样本量以加快计算
        if X.shape[0] > sample_size:
            X_sample = X[:sample_size]
        else:
            X_sample = X

        # 为每个标签创建SHAP摘要图
        for idx, (model, label) in enumerate(zip(self.models, label_names)):
            try:
                # 创建解释器
                explainer = shap.TreeExplainer(model)

                # 计算SHAP值
                shap_values = explainer.shap_values(X_sample)

                # 处理二分类情况（取正类的SHAP值）
                if isinstance(shap_values, list) and len(shap_values) == 2:
                    shap_values = shap_values[1]

                # 创建图表
                plt.figure(figsize=(10, 8))
                shap.summary_plot(
                    shap_values,
                    X_sample,
                    feature_names=feature_names,
                    plot_type="dot",
                    show=False
                )

                # 设置SHAP图标题字体大小
                plt.title(f"SHAP特征重要性 - {label}", fontsize=SHAP_TITLE_FONT_SIZE)

                # 设置SHAP图特征名称字体大小
                plt.rcParams.update({'font.size': SHAP_FEATURE_FONT_SIZE})

                plt.tight_layout()
                plt.savefig(f"{save_dir}/SHAP_{label}.png", dpi=300, bbox_inches='tight')
                plt.close()
            except Exception as e:
                print(f"生成{label}的SHAP图时出错: {str(e)}")

        # 生成综合特征重要性图
        self._plot_combined_importance(X_sample, feature_names, label_names, save_dir)

    def _plot_combined_importance(self, X, feature_names, label_names, save_dir):
        total_importance = np.zeros(len(feature_names))

        for idx, model in enumerate(self.models):
            try:
                # 获取特征重要性
                importance = model.feature_importances_
                total_importance += importance
            except:
                # 如果无法获取，使用SHAP值计算
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

        # 绘制横向条形图
        plt.figure(figsize=(12, 0.5 * len(feature_names)))
        plt.barh(importance_df["特征"], importance_df["重要性"], color="teal")

        # 设置综合特征重要性图坐标轴标题字体大小
        plt.xlabel("平均特征重要性", fontsize=COMBINED_AXIS_FONT_SIZE)

        # 设置综合特征重要性图标题字体大小
        plt.title("综合特征重要性（四标签平均）", fontsize=COMBINED_TITLE_FONT_SIZE)

        # 设置综合特征重要性图特征名称字体大小
        plt.tick_params(axis='y', labelsize=COMBINED_FEATURE_FONT_SIZE)

        # 自动调整坐标范围以适应更大的数值
        plt.xlim(0, importance_df["重要性"].max() * 1.2)

        plt.tight_layout()
        plt.savefig(f"{save_dir}/综合特征重要性.png", dpi=300, bbox_inches='tight')
        plt.close()


# ========== 模型评估 ==========
def evaluate_model(model, X, y, label_names):
    """执行5折交叉验证并返回指标"""
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    metrics = {
        'Hamming Loss': [],
        'Accuracy': [],
        'Macro F1': [],
        'Micro F1': [],
        'Precision (Macro)': [],
        'Recall (Macro)': [],
        'Jaccard Score': []
    }

    for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # 克隆模型以避免参数污染
        cloned_model = clone(model)
        cloned_model.fit(X_train, y_train)
        y_pred = cloned_model.predict(X_test)

        # 计算指标
        metrics['Hamming Loss'].append(hamming_loss(y_test, y_pred))
        metrics['Accuracy'].append(accuracy_score(y_test, y_pred))
        metrics['Macro F1'].append(f1_score(y_test, y_pred, average='macro'))
        metrics['Micro F1'].append(f1_score(y_test, y_pred, average='micro'))
        metrics['Precision (Macro)'].append(precision_score(y_test, y_pred, average='macro'))
        metrics['Recall (Macro)'].append(recall_score(y_test, y_pred, average='macro'))
        metrics['Jaccard Score'].append(jaccard_score(y_test, y_pred, average='samples'))

        # 打印分类报告
        print(f"\nFold {fold} 分类报告:")
        print(classification_report(y_test, y_pred, target_names=label_names))

    # 计算平均指标
    return {k: np.mean(v) for k, v in metrics.items()}


# ========== 主程序 ==========
def main():
    # 加载数据
    data_path = "\标准化.xlsx"
    X, y, feature_names, label_names = load_data(data_path)

    # 分割数据集（用于最终评估和可视化）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,
        random_state=42
    )

    # 参数网格搜索
    param_grid = {
        'n_estimators': [100],
        'max_depth': [10]
    }

    best_score = -1
    best_params = None

    # 网格搜索
    for n_est in param_grid['n_estimators']:
        for max_depth in param_grid['max_depth']:
            print(f"\n正在尝试参数组合: n_estimators={n_est}, max_depth={max_depth}")

            model = BR_RF(
                n_estimators=n_est,
                max_depth=max_depth,
                random_state=42
            )

            # 交叉验证评估
            metrics = evaluate_model(model, X_train, y_train, label_names)
            current_score = metrics['Macro F1']

            print(f"当前得分: {current_score:.4f}")
            print("评估指标:", {k: f"{v:.4f}" for k, v in metrics.items()})

            if current_score > best_score:
                best_score = current_score
                best_params = {'n_estimators': n_est, 'max_depth': max_depth}
                print("发现新的最佳参数组合!")

    # 训练最终模型
    print("\n=== 训练最终模型 ===")
    print(f"最佳参数: {best_params}")
    final_model = BR_RF(**best_params, random_state=42)
    final_model.fit(X_train, y_train)

    # 在测试集上评估
    y_pred = final_model.predict(X_test)
    print("\n测试集分类报告:")
    print(classification_report(y_test, y_pred, target_names=label_names))

    # SHAP可视化
    print("\n生成SHAP可视化...")
    final_model.plot_shap_analysis(
        X_test[:100],  # 使用前100个测试样本加快计算
        feature_names=feature_names,
        label_names=label_names,
        save_dir="BR_RF_SHAP_Results1010"
    )
    print("可视化结果已保存至 BR_RF_SHAP_Results 目录")


if __name__ == "__main__":
    main()