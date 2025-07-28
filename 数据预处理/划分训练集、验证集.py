# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.model_selection import train_test_split

# 读取Excel文件
file_path = "D:\\研二2\\论文撰写\\标准化、填充缺失值后的数据.xlsx"  # 修改为你的Excel文件路径
df = pd.read_excel(file_path)

# 分离数据
id_col = df.iloc[:, 0]        # 第一列是ID
X = df.iloc[:, 1:-4]          # 中间所有列作为特征X
y = df.iloc[:, -4:]           # 最后四列作为目标y

# 划分数据集(保持ID与数据对应)
X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
    X, y, id_col,
    test_size=0.2,
    random_state=42,
    stratify=y.iloc[:, 0]     # 仅用第一个目标变量分层
)

# 重组数据集(合并ID+特征+目标)
train_data = pd.concat([id_train.reset_index(drop=True),
                       X_train.reset_index(drop=True),
                       y_train.reset_index(drop=True)], axis=1)

test_data = pd.concat([id_test.reset_index(drop=True),
                      X_test.reset_index(drop=True),
                      y_test.reset_index(drop=True)], axis=1)

# 保存结果
train_data.to_excel("D:\\研二2\\论文撰写\\训练集.xlsx", index=False)
test_data.to_excel("D:\\研二2\\论文撰写\\测试集.xlsx", index=False)

print("划分完成！")
print(f"训练集样本数: {len(train_data)}")
print(f"测试集样本数: {len(test_data)}")
print(f"输出文件: train_dataset.xlsx 和 test_dataset.xlsx")