import matplotlib

matplotlib.use('TkAgg')
import pandas as pd
import matplotlib.pyplot as plt
from tkinter import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
LOG_LINE_NUM = 0


class mclass:
    def __init__(self, window):
        self.window = window
        self.box = Entry(window)
        self.button = Button(window, text="check", command=self.plot)
        self.box.pack()
        self.button.pack()
        frame1 = Frame(window, bg="#ffffff")
        frame1.place(x=5, y=50, width=690, height=700)

    def plot(self):
        origdata = pd.read_csv("../../resource/csv/CM1.csv")
        origdata[:10]
        print(len(origdata.columns))

        # 选取数据（获取数据的列）
        k = []
        for i in range(0, len(origdata.columns)):
            k.append(origdata.columns.values[i])  # print(origdata.columns.values[1])#列名数据提取样式
        # print(k)
        data = origdata.copy()

        # 训练数据
        k1, k2, k3 = k[2], k[3], k[4]
        X = data[[k1, k2, k3]]
        y = data[k[37]]
        print('Classes:')
        print(y.unique(), '\n\n\n')

        y[y == 'b\'N\''] = 1
        y[y == 'b\'Y\''] = 2
        y[y == 'unknown'] = 0

        fig = plt.figure(dpi=128, figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        versicolor = y == 1
        virginica = y == 2

        # 绘制
        ax.scatter(X[k1][versicolor], X[k2][versicolor], X[k3][versicolor], c='r')
        ax.scatter(X[k1][virginica], X[k2][virginica], X[k3][virginica], c='c')
        ax.set
        ax.set_xlabel('X', fontsize=10)
        ax.set_ylabel('Y', fontsize=10)
        ax.set_zlabel('Z', fontsize=10)

        canvas = FigureCanvasTkAgg(fig, master=self.window)
        canvas.get_tk_widget().pack()
        canvas.draw()


window = Tk()
start = mclass(window)
window.mainloop()
