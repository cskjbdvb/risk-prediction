
import matplotlib
matplotlib.use('Agg')  # 强制使用非交互式后端

import matplotlib.pyplot as plt
from matplotlib import font_manager

# === 中文字体设置 ===
font_path = 'C:\\Windows\\Fonts\\msyhl.ttc'  # 替换为你的实际路径
font_manager.fontManager.addfont(font_path)
plt.rcParams['font.family'] = font_manager.FontProperties(fname=font_path).get_name()
plt.rcParams['axes.unicode_minus'] = False

# === 数据准备 ===
categories = ['类别一', '类别二', '类别三', '类别四']
values = [23, 45, 12, 38]

# === 绘图 ===
fig, ax = plt.subplots()
bars = ax.bar(categories, values)

# 设置标题和标签
ax.set_title('中文字体测试图')
ax.set_xlabel('类别')
ax.set_ylabel('数值')

# 添加数据标签
for bar in bars:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2.,
        height,
        f'{height}',
        ha='center',
        va='bottom'
    )

# 保存图像（替代plt.show()）
plt.savefig('chinese_font_test.png', bbox_inches='tight', dpi=300)
print("图像已保存为 chinese_font_test.png")