import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# ===================== 必须修改的配置 =====================
INPUT_FILE = "D:\\研二2\\删除部分无关特征.xlsx"  # 您的Excel文件路径
ID_COL = '响应者序列号'  # ID列名称
LABEL_COLS = ['慢阻肺', '中风', '冠心病','糖尿病']  # 四个标签列名
CATEGORICAL_THRESHOLD = 8  # 唯一值超过此数视为连续型

def load_data():
    """加载数据并显示基本信息"""
    df = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME)
    print("\n=== 数据加载成功 ===")
    print(f"总样本数: {len(df)} | 总特征数: {len(df.columns) - 5}")
    return df


def strict_feature_identification(df):
    """严格的特征类型识别"""
    feature_types = {}
    all_features = [col for col in df.columns if col not in [ID_COL] + LABEL_COLS]

    for feature in all_features:
        # 优先使用手动指定的类型
        if feature in MANUAL_TYPES:
            feature_types[feature] = MANUAL_TYPES[feature]
            continue

        # 获取非空唯一值
        unique_vals = df[feature].dropna().unique()
        num_unique = len(unique_vals)

        # 严格类型判断逻辑
        if num_unique == 2 and set(unique_vals) == {0, 1}:
            feature_types[feature] = "binary"  # 已经是0/1编码的二分类
        elif num_unique == 2:
            feature_types[feature] = "binary"  # 二分类但未编码
        elif num_unique <= 8:  # 假设类别数≤5为分类变量
            feature_types[feature] = "multi"  # 多分类
        elif pd.api.types.is_numeric_dtype(df[feature]):
            feature_types[feature] = "continuous"  # 连续型
        else:
            feature_types[feature] = "multi"  # 默认作为多分类

    return feature_types


def process_data(df, feature_types):
    """执行特征处理"""
    # 分类特征
    binary_feats = [f for f, t in feature_types.items() if t == "binary"]
    multi_feats = [f for f, t in feature_types.items() if t == "multi"]
    continuous_feats = [f for f, t in feature_types.items() if t == "continuous"]

    print("\n=== 特征类型确认 ===")
   
    # 创建处理管道
    preprocessor = ColumnTransformer(
        transformers=[
            ('continuous', StandardScaler(), continuous_feats),
            ('multi_cat', OneHotEncoder(drop='first', sparse_output=False), multi_feats)
        ],
        remainder="passthrough"  # 保持其他列不变
    )

    # 应用转换
    features_to_process = df.drop(columns=[ID_COL] + LABEL_COLS)
    processed_array = preprocessor.fit_transform(features_to_process)

    # 构建新特征名称
    new_columns = continuous_feats.copy()  # 连续型特征名不变

    # 添加独热编码后的列名
    if multi_feats:
        ohe = preprocessor.named_transformers_['multi_cat']
        for i, col in enumerate(multi_feats):
            categories = ohe.categories_[i][1:]  # 使用drop='first'
            new_columns.extend([f"{col}_{cat}" for cat in categories])

    # 添加二分类特征名
    new_columns.extend(binary_feats)

    # 创建最终DataFrame
    processed_df = pd.DataFrame(processed_array, columns=new_columns)

    # 添加ID和标签列
    processed_df[ID_COL] = df[ID_COL].values
    for label in LABEL_COLS:
        processed_df[label] = df[label].values

    # 优化列顺序
    final_order = (
            [ID_COL] + LABEL_COLS +
            binary_feats + continuous_feats +
            [c for c in new_columns if c not in binary_feats + continuous_feats]
    )

    return processed_df[final_order]


def save_results(processed_df):
    """保存处理结果"""
    output_file = "标准化.xlsx"
    processed_df.to_excel(output_file, index=False)

    print("\n=== 处理结果 ===")
    print(f"结果文件已保存: {output_file}")



if __name__ == "__main__":
    # 执行流程
    df = load_data()
    feature_types = strict_feature_identification(df)
    final_data = process_data(df, feature_types)
    save_results(final_data)
    print("\n处理完成！所有特征已正确转换")



