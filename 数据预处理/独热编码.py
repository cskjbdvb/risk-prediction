import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# ===================== 必须修改的配置 =====================
INPUT_FILE = 'D:\\研二2\\论文撰写\\数据合并\\箱线图.xlsx' # 您的Excel文件路径
SHEET_NAME = "Sheet1"  # 工作表名称
ID_COL = 'SEQN'  # ID列名称
LABEL_COLS = ['DIQ010 - 医生告诉你有糖尿病', 'MCQ160c - 曾被告知自己患有冠心病', 'MCQ160f - 曾被告知你中风','BPQ020 - 曾说过你得高血压']  # 四个标签列名
#CATEGORICAL_THRESHOLD = 8  # 唯一值超过此数视为连续型]  # 四个标签列名

# 在这里手动指定需要独热编码的多分类特征列名
MULTI_CAT_FEATURES = ['DMDEDUC2 - 教育水平 - 成人 20+',
                                    'DMDHHSIZ - 家庭总人数',
                                    'DMDHRAGE-HH筛查时家庭参考人的年龄。',
                                    'DMDMARTL-婚姻状况',
                                    'RIDRETH1 - 种族/西班牙裔',
                                    'OCD150 - 上周完成的工作类型',
                                    'OCD30G-哪个职业做的时间最长',
                                    'HSD010-一般健康状况'
    # 替换为您的第二个多分类特征名
    # ...添加更多需要处理的特征
]


def load_data():
    """加载数据并显示基本信息"""
    df = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME)
    print("\n=== 数据加载成功 ===")
    print(f"总样本数: {len(df)} | 总特征数: {len(df.columns) - 5}")
    print("\n前3行数据预览:")
    print(df.head(3))
    return df


def onehot_encode_specific_features(df):
    """只对指定的多分类特征进行独热编码"""
    # 验证指定的特征是否存在
    missing_features = [f for f in MULTI_CAT_FEATURES if f not in df.columns]
    if missing_features:
        raise ValueError(f"以下特征不存在: {missing_features}")

    # 复制原始数据（保留所有原始列）
    processed_df = df.copy()

    # 对每个指定特征进行独热编码
    for feature in MULTI_CAT_FEATURES:
        # 创建独热编码器（不丢弃任何类别）
        encoder = OneHotEncoder(sparse_output=False, drop=None)
        encoded_data = encoder.fit_transform(processed_df[[feature]])

        # 获取新列名（特征名_类别名）
        new_columns = [f"{feature}_{str(cat)}" for cat in encoder.categories_[0]]

        # 将编码结果转换为DataFrame
        encoded_df = pd.DataFrame(encoded_data, columns=new_columns)

        # 合并到结果DataFrame
        processed_df = pd.concat([processed_df, encoded_df], axis=1)

        # 移除原始特征列
        processed_df.drop(columns=[feature], inplace=True)

    return processed_df


def save_results(processed_df):
    """保存处理结果"""
    output_file = 'D:\\研二2\\论文撰写\\数据合并\\独热编码.xlsx'
    processed_df.to_excel(output_file, index=False)

    print("\n=== 处理结果 ===")
    print(f"📁 结果文件已保存: {output_file}")
    print(f"🔄 形状变化: {len(df)}行 × {len(df.columns)}列 → {processed_df.shape}")
    print("\n处理后3行数据预览:")
    print(processed_df.head(3))


if __name__ == "__main__":
    print("=== 开始执行 ===")
    df = load_data()

    print("\n=== 正在处理 ===")
    print(f"将对以下特征进行独热编码: {MULTI_CAT_FEATURES}")

    final_data = onehot_encode_specific_features(df)
    save_results(final_data)

    print("\n✅ 处理完成！只有指定特征被转换，其他特征保持不变")
