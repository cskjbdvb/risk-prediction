import pandas as pd
import numpy as np
#from skmultilearn.feature_selection import MIFS
import mifs

from sklearn.model_selection import train_test_split
from skmultilearn.adapt import MLkNN
from sklearn.metrics import hamming_loss, accuracy_score, f1_score
from time import time


# 1. 数据加载
def load_data(file_path):
    df = pd.read_excel(file_path)
    ids = df['ID'].values
    labels = df.iloc[:, -4:].values.astype(int) 
    features = df.iloc[:, 1:-4].values 
    feature_names = df.columns[1:-4].tolist()
    return ids, features, labels, feature_names


# 2. 特征选择与评估
def evaluate_mifs(X, y, k_features, alpha_values):
    """评估不同参数下的MIFS表现"""
    # 划分训练测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    results = []
    for alpha in alpha_values:
        for k in k_features:
            print(f"\n正在评估 alpha={alpha}, k={k}...")

            # 特征选择
            start_time = time()
            mifs = mifs(
                k=k,  # 选择前k个特征
                alpha=alpha,  # 标签相关性权重
                sigma=0.1,  
                n_jobs=-1 
            )
            X_train_selected = mifs.fit_transform(X_train, y_train)
            X_test_selected = mifs.transform(X_test)
            selection_time = time() - start_time

            # 训练分类器评估效果
            classifier = MLkNN(k=5)
            classifier.fit(X_train_selected, y_train)
            y_pred = classifier.predict(X_test_selected)

            # 计算多个指标
            metrics = {
                'alpha': alpha,
                'k_features': k,
                'selection_time': round(selection_time, 2),
                'hamming_loss': hamming_loss(y_test, y_pred),
                'accuracy': accuracy_score(y_test, y_pred),
                'macro_f1': f1_score(y_test, y_pred, average='macro'),
                'micro_f1': f1_score(y_test, y_pred, average='micro')
            }
            results.append(metrics)

            print(f"特征选择耗时: {selection_time:.2f}s")
            print(f"Hamming Loss: {metrics['hamming_loss']:.4f}")
            print(f"Macro F1: {metrics['macro_f1']:.4f}")

    return pd.DataFrame(results)


# 3. 保存结果
def save_results(results_df, selected_features, output_file):
    with pd.ExcelWriter(output_file) as writer:
        results_df.to_excel(writer, sheet_name='参数评估', index=False)
        pd.DataFrame({'selected_features': selected_features}).to_excel(
            writer, sheet_name='选中特征', index=False)


# 主流程
if __name__ == "__main__":
    # 参数设置
    input_file = "标准化.xlsx"  # 替换为你的Excel文件路径
    output_file = "\MIFS.xlsx"

    # MIFS参数网格
    alpha_values = [0.1, 0.5, 1.0]  # 标签相关性权重
    k_features = [50, 100, 150]  # 选择特征数量

    # 1. 加载数据
    ids, X, y, feature_names = load_data(input_file)
    print(f"原始数据形状: {X.shape}, 标签形状: {y.shape}")

    # 2. 特征选择评估
    results_df = evaluate_mifs(X, y, k_features, alpha_values)

    # 3. 选择最佳参数（根据macro_f1）
    best_params = results_df.loc[results_df['macro_f1'].idxmax()]
    print("\n最佳参数组合:")
    print(best_params)

    # 4. 用最佳参数重新训练
    final_mifs = mifs(
        k=int(best_params['k_features']),
        alpha=best_params['alpha'],
        sigma=0.1,
        n_jobs=-1
    )
    X_selected = final_mifs.fit_transform(X, y)
    selected_indices = final_mifs.get_support(indices=True)
    selected_features = [feature_names[i] for i in selected_indices]

    # 5. 保存结果
    save_results(results_df, selected_features, output_file)
    print(f"\n结果已保存到 {output_file}")
    print(f"最终选择特征数: {len(selected_features)}")
    print("前10个重要特征:", selected_features[:10])
