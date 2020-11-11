import matplotlib.pyplot as plt
import numpy as np
fig = plt.figure(figsize=(12, 8),
                 facecolor='lightyellow'
                 )

# 创建 3D 坐标系
ax = fig.gca(fc='whitesmoke',
             projection='3d'
             )

# 二元函数定义域
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)

# -------------------------------- 绘制 3D 图形 --------------------------------
# 平面 z=3 的部分

# 平面 z=2y 的部分
# ax.plot_surface(X,
#                 Y=Y,
#                 Z=Y * 2,
#                 color='y',
#                 alpha=0.6
#                 )
# # 平面 z=-2y + 10 部分
# ax.plot_surface(X=X,
#                 Y=Y,
#                 Z=-Y * 2 + 10,
#                 color='r',
#                 alpha=0.7
#                 )
# --------------------------------  --------------------------------


u = np.linspace(0,2*np.pi,50)  # 把圆分按角度为50等分
h = np.linspace(-30,30,100)        # 把高度1均分为20份
x = np.outer(np.sin(u),np.ones(len(h)))  # x值重复20次
y = np.outer(np.cos(u),np.ones(len(h)))  # y值重复20次
z = np.outer(np.ones(len(u)),h)   # x，y 对应的高度

# Plot the surface
ax.plot_surface(x*2, y*2, z,)
ax.plot_surface(X,
                Y,
                Z=X+2*Y-2,
                color='g'
                )
# 设置坐标轴标题和刻度
ax.set(xlabel='X',
       ylabel='Y',
       zlabel='Z',

       )

# 调整视角
ax.view_init(elev=80,  # 仰角
             azim=40  # 方位角
             )

# 显示图形
plt.show()
a = np.zeros([3])
b = a
b =a+ b  # 和 b = b + 3
