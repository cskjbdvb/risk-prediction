import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

import matplotlib
matplotlib.use('TkAgg')

plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "Arial Unicode MS"]
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

# 读取Excel文件
file_path = "26397.xlsx"
df = pd.read_excel(file_path, index_col=0)
print(f"成功读取数据：{df.shape[0]}行，{df.shape[1]}列")


colors = sns.color_palette('tab20', n_colors=len(df.index))

plt.figure(figsize=(14, 9))

boxprops = dict(linewidth=1.5)
medianprops = dict(linewidth=2.0, color='black')
whiskerprops = dict(linestyle='-')
capprops = dict(linewidth=1.5)

positions = range(1, len(df.index) + 1)
boxplot = plt.boxplot(
    df.T.values,
    positions=positions,
    patch_artist=True,
    boxprops=boxprops,
    medianprops=medianprops,
    whiskerprops=whiskerprops,
    capprops=capprops,
    widths=0.6,
    showfliers=False
)


for patch, color in zip(boxplot['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)

# 设置图表标题和轴标签
plt.title('', fontsize=18)
plt.xlabel('', fontsize=16)
plt.ylabel('AUC', fontsize=16)

# 设置x轴刻度和标签
plt.xticks(positions, df.index, rotation=0, ha='center', fontsize=11)
# 添加网格线
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()

# 保存图表
try:
    plt.savefig('hamming_loss_boxplot.png', dpi=300, bbox_inches='tight')
    print("图表已保存为 'hamming_loss_boxplot.png'")
except Exception as e:
    print(f"保存图表时出错: {e}")

# 显示图形
try:
    plt.show()
except Exception as e:
    print(f"显示图表时出错: {e}")
    print("请查看保存的图片文件")