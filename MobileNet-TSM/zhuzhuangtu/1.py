import matplotlib.pyplot as plt
import numpy as np

# 这两行代码解决 plt 中文显示的问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 输入统计数据
waters = ('Crowd violence', 'Hockey', 'RWF-2000', 'Expanded dataset')
Two_cascade_TSM = [1.00, 0.50, 1.00, 0.20]
Partial_TSM = [1.00, 1.00, 3.00, 1.20]

bar_width = 0.3  # 条形宽度
index_male = np.arange(len(waters))  # 男生条形图的横坐标
index_female = index_male + bar_width  # 女生条形图的横坐标

# 使用两次 bar 函数画出两组条形图
plt.bar(index_male, height=Two_cascade_TSM, width=bar_width, color='#5B9DD5', label='Two-cascade TSM')
plt.bar(index_female, height=Partial_TSM, width=bar_width, color='#ED7D31', label='Partial TSM')

plt.legend()  # 显示图例
plt.xticks(index_male + bar_width/2, waters)  # 让横坐标轴刻度显示 waters 里的饮用水， index_male + bar_width/2 为横坐标轴刻度的位置
# plt.ylabel('购买量')  # 纵坐标轴标题
# plt.title('购买饮用水情况的调查结果')  # 图形标题

plt.show()
