import numpy as np
from tkinter import *
from tkinter import messagebox
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
# 创建画布需要的库
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

root = Tk()
root.title("tkinter + Matplotlib")
root.geometry('700x750')

Label(root, text='tkinter & Matplotlib动态示例').place(x=0, y=0, width=700, height=50)
# 创建一个容器, 没有画布时的背景
frame1 = Frame(root, bg="#ffffff")
frame1.place(x=5, y=50, width=690, height=700)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
fig = plt.figure(figsize=(6.5, 7), edgecolor='blue')
ax = Axes3D(fig)
# 定义刻度
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.set_zlim(0, 100)
canvas = FigureCanvasTkAgg(fig, master=frame1)
canvas.draw()
# 显示画布
canvas.get_tk_widget().place(x=0, y=0)

i = 0
# 定义存储坐标的空数组
x = []
y = []
z = []


# 抛物线动态绘制函数
def drawImg():
    global i
    i += 1
    global ax
    ax.clear()
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_zlim(0, 100)
    global x
    global y
    global z
    t = i * 0.1
    dtax = 20 * t * np.sin(0.25 * np.pi)
    x.append(dtax)
    dtay = 20 * t * np.cos(0.25 * np.pi)
    y.append(dtay)
    dtaz = 100 - t ** 2 * 0.5 * 10
    z.append(dtaz)
    ax.plot(x, y, z)
    canvas.draw()
    global afterHandler
    afterHandler = root.after(100, drawImg)


drawImg()


def on_closing():
    root.after_cancel(afterHandler)
    answer = messagebox.askokcancel("退出", "确定退出吗?")
    if answer:
        plt.close('all')
        root.destroy()
    else:
        root.after(1000, drawImg)


root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()

