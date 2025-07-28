import pandas as pd
import os

# 设置文件路径
folder_path = 'D:\\数据合并'  # 替换为实际文件夹路径
output_file = 'D:\\研二2\\论文撰写\\实验数据合并.xlsx'  # 输出文件名

# 获取所有Excel文件
excel_files = [f for f in os.listdir(folder_path) if f.endswith('.xlsx') or f.endswith('.xls')]

# 初始化一个空DataFrame用于存储合并结果
merged_data = pd.DataFrame()

# 循环处理每个文件
for file in excel_files:
    file_path = os.path.join(folder_path, file)
    df = pd.read_excel(file_path)

    # 确保ID列名一致，假设列名为'ID'
    if 'SEQN_new' not in df.columns:
        print(f"文件 {file} 中没有SEQN_new列，跳过")
        continue

    if merged_data.empty:
        merged_data = df
    else:
        # 根据ID合并数据
        merged_data = pd.merge(merged_data, df, on='SEQN_new', how='outer')

# 保存合并结果
merged_data.to_excel(output_file, index=False)
print(f"合并完成，结果已保存到 {output_file}")