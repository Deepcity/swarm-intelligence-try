import matplotlib.pyplot as plt

# 定义点
set1 = [(2, 12), (6, 8), (6, 16), (10, 12)]
set2 = [(5, 26), (5, 32), (6, 26), (6, 32), (7, 26), (7, 32), 
         (8, 26), (8, 32), (9, 26), (9, 32), (10, 26), (10, 32), 
         (11, 26), (11, 32), (5, 27), (11, 27), (5, 28), (11, 28), 
         (5, 29), (11, 29), (5, 30), (11, 30), (5, 31), (11, 31)]
set3 = [(27, 32), (27, 38), (28, 32), (28, 38), (29, 32), 
         (29, 38), (30, 32), (30, 38), (31, 32), (31, 38), 
         (32, 32), (32, 38), (33, 32), (33, 38), (27, 33), 
         (33, 33), (27, 34), (33, 34), (27, 35), (33, 35), 
         (27, 36), (33, 36), (27, 37), (33, 37)]

# 分离坐标
x1, y1 = zip(*set1)
x2, y2 = zip(*set2)
x3, y3 = zip(*set3)

# 创建绘图
plt.figure(figsize=(10, 6))

# 绘制点
plt.scatter(x1, y1, color='red', label='Set 1', marker='o')
plt.scatter(x2, y2, color='blue', label='Set 2', marker='x')
plt.scatter(x3, y3, color='green', label='Set 3', marker='s')

# 设置图形属性
plt.title('Scatter Plot of Points')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid(True)
plt.axhline(0, color='black',linewidth=0.5, ls='--')
plt.axvline(0, color='black',linewidth=0.5, ls='--')
plt.xlim(0, 35)
plt.ylim(0, 40)
plt.legend()

# 显示图形
plt.show()
