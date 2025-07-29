import numpy as np
from sklearn.preprocessing import StandardScaler
import shap
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.ensemble import RandomForestClassifier
import warnings
import matplotlib
matplotlib.use('TkAgg')


# 中文显示配置
font_path = 'C:\\Windows\\Fonts\\msyhl.ttc'
font_manager.fontManager.addfont(font_path)
plt.rcParams['font.family'] = font_manager.FontProperties(fname=font_path).get_name()
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore')
shap.initjs()


def load_data(file_path):
    data = pd.read_excel(file_path)
    X = data.iloc[:, 1:-1].values
    y = data.iloc[:, -1].values
    feature_names = data.columns[1:-1].tolist()
    return X, y, feature_names


def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model.fit(X_scaled, y_train)
    return model, scaler


def visualize_shap(model, scaler, X, feature_names, sample_idx, y_true):
    try:
        sample = X[sample_idx]
        sample_scaled = scaler.transform(sample.reshape(1, -1))

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(sample_scaled)

        feature_labels = [f"{name} = {value:.2f}" if isinstance(value, (float, np.floating))
                          else f"{name} = {value}"
                          for name, value in zip(feature_names, sample)]

        shap_explanation = shap.Explanation(
            values=shap_values[1][0],
            base_values=explainer.expected_value[1],
            data=sample,
            feature_names=feature_names
        )

        # 绘制图形
        plt.figure(figsize=(100, 110))
        shap.plots.force(
            explainer.expected_value[1],
            shap_values[1][0],
            feature_names=feature_labels,
            matplotlib=True,
            show=False,
            text_rotation=0
        )

        proba = model.predict_proba(sample_scaled)[0][1]
        plt.title(
            f"样本索引: {sample_idx} | 真实类别: {y_true[sample_idx]}\n"
            f"预测概率 = {proba:.4f} | 基准值 = {explainer.expected_value[1]:.4f}",
            pad=25, fontsize=12
        )
        plt.tight_layout()

        output_path = f"SHAP_样本_{sample_idx}7.png"
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()

        print(f"\n分析结果已保存至: {output_path}")
        print("特征值显示格式示例:", feature_labels[:5])  # 显示前3个特征格式

    except Exception as e:
        print(f"错误: {str(e)}")
        print("建议检查特征值格式或图形尺寸")


def main():
    # 配置参数
    DATA_PATH = "标准化.xlsx"
    SAMPLE_IDX = 5

    # 加载数据
    X, y, feature_names = load_data(DATA_PATH)

    # 分割数据
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)

    # 训练模型
    print("训练模型中...")
    model, scaler = train_model(X_train, y_train)
    print(f"模型准确率: {model.score(scaler.transform(X_test), y_test):.3f}")

    # 解释指定样本
    print(f"\n分析测试集样本索引 {SAMPLE_IDX}...")
    if SAMPLE_IDX < len(X_test):
        visualize_shap(model, scaler, X_test, feature_names, SAMPLE_IDX, y_test)
    else:
        print(f"错误: 索引 {SAMPLE_IDX} 超出测试集范围(0-{len(X_test) - 1})")


if __name__ == "__main__":
    main()