import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# ---------------------------
# 1. 数据加载
# ---------------------------
def load_data(file_path):
    """加载Excel数据"""
    data = pd.read_excel(file_path)
    return data

# ---------------------------
# 2. 特征选择 - 互信息
# ---------------------------
def select_features_by_mutual_info(data, target_column, top_k=10):
    """
    使用互信息选择特征
    :param data: 包含特征和标签的DataFrame
    :param target_column: 标签列名
    :param top_k: 选择的特征数量
    :return: 重要特征的名称列表
    """
    # 分离特征和标签
    X = data.drop(columns=[data.columns[0], target_column])  # 去掉ID列和目标列
    y = data[target_column]

    # 计算互信息
    mutual_info = mutual_info_classif(X, y)
    mutual_info = pd.Series(mutual_info, index=X.columns)
    mutual_info = mutual_info.sort_values(ascending=False)

    # 选择前k个特征
    selected_features = mutual_info.head(top_k).index.tolist()

    return selected_features, mutual_info

# ---------------------------
# 3. 特征重要性可视化
# ---------------------------
def plot_feature_importance(mutual_info, top_k=10):
    """绘制特征重要性条形图"""
    plt.figure(figsize=(12, 8))
    mutual_info.head(top_k).plot(kind='barh')
    plt.title(f"Top {top_k} Features by Mutual Information")
    plt.xlabel("Mutual Information")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300)
    plt.close()

# ---------------------------
# 4. 特征选择后的模型训练与评估
# ---------------------------
def train_and_evaluate_model(data, selected_features, target_column):
    """
    使用选择的特征训练模型并评估性能
    :param data: 包含特征和标签的DataFrame
    :param selected_features: 选择的特征列表
    :param target_column: 标签列名
    :return: 训练后的模型和评估结果
    """
    # 分离特征和标签
    X = data[selected_features]
    y = data[target_column]

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 初始化模型
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    # 训练模型
    model.fit(X_train, y_train)

    # 预测
    y_pred = model.predict(X_test)

    # 评估
    report = classification_report(y_test, y_pred)
    print("模型评估报告:")
    print(report)

    return model, report

# ---------------------------
# 5. 保存结果
# ---------------------------
def save_results(data, mutual_info, selected_features, output_path):
    """保存特征重要性和选择的特征"""
    # 创建结果DataFrame
    results = pd.DataFrame({
        'Feature': mutual_info.index,
        'Mutual Information': mutual_info.values
    })

    # 保存结果
    results.to_excel(output_path, index=False)
    print(f"特征重要性结果已保存到 {output_path}")

# ---------------------------
# 6. 执行代码
# ---------------------------
if __name__ == "__main__":
    # 文件路径配置
    file_path = "D:\\研二2\\论文撰写\\数据合并\\标准化、缺失值\\train_resampled_manual_mlsmote.xlsx"  # 替换为实际路径
    target_column = "Label_4"  # 替换为实际的疾病标签列名
    output_path = "feature_importance_results.xlsx"

    # 加载数据
    data = load_data(file_path)

    # 特征选择
    selected_features, mutual_info = select_features_by_mutual_info(data, target_column, top_k=10)

    # 可视化特征重要性
    plot_feature_importance(mutual_info, top_k=10)

    # 保存结果
    save_results(data, mutual_info, selected_features, output_path)

    # 训练和评估模型
    model, report = train_and_evaluate_model(data, selected_features, target_column)

    print("\n特征选择和模型训练完成！")
    print(f"选择的特征: {selected_features}")
    print(f"特征重要性结果已保存到 {output_path}")
    print(f"特征重要性图已保存为 feature_importance.png")