import matplotlib.pyplot as plt

ax = []  # 定义一个 x 轴的空列表用来接收动态的数据
ay = []  # 定义一个 y 轴的空列表用来接收动态的数据
 # 开启一个画图的窗口

def point(y):
    plt.figure(1)
    ax.append(len(ax))  # 添加 i 到 x 轴的数据中
    ay.append(y)  # 添加 i 的平方到 y 轴的数据中
    plt.plot(ax, ay)  # 画出当前 ax 列表和 ay 列表中的值的图形
    plt.show()

def plot(x, y):
    # 遍历0-99的值
    ax.append(x)  # 添加 i 到 x 轴的数据中
    ay.append(y)  # 添加 i 的平方到 y 轴的数据中
    plt.clf()  # 清除之前画的图
    plt.plot(ax, ay)  # 画出当前 ax 列表和 ay 列表中的值的图形



if __name__=="__main__":
    point(2)
    point(2)
    point(2)
