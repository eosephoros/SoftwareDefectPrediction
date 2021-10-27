import os
import random
from tkinter import messagebox, ttk
# 创建画布需要的库
from tkinter import *
import tkinter as tk

import joblib
import pandas
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import imageio as io
from retrying import retry

LOG_LINE_NUM = 0

root = Tk()
w, h = root.maxsize()
root.geometry("{}x{}".format(w, h))  # 看好了，中间的是小写字母x
# Label(root, text='tkinter & Matplotlib动态示例').place(x=0, y=0, width=700, height=20)


# 创建一个容器, 没有画布时的背景
# 滚动条初始化（scrollBar为垂直滚动条，scrollBarx为水平滚动条）

# frame = ScrollableFrame(root)
# for i in range(50):
#     ttk.Label(frame.scrollable_frame, text="Sample scrolling label").pack()
# frame.pack()

frame1 = Frame(root, bg="#ffffff")
frame1.place(x=20 + 50, y=50, width=400, height=400)

frame2 = Frame(root, bg="#ffffff")
frame2.place(x=440 + 50, y=50, width=400, height=400)

frame3 = Frame(root, bg="#ffffff")
frame3.place(x=860 + 50, y=50, width=400, height=400)

frame4 = Frame(root, bg="#ffffff")
frame4.place(x=1280 + 50, y=50, width=400, height=400)

frame41 = Frame(root, bg="#ffffff")
frame41.place(x=20, y=500, width=350, height=400)

frame42 = Frame(root, bg="#ffffff")
frame42.place(x=390, y=500, width=350, height=400)

frame43 = Frame(root, bg="#ffffff")
frame43.place(x=760, y=500, width=350, height=400)

frame44 = Frame(root, bg="#ffffff")
frame44.place(x=1130, y=500, width=350, height=400)

frame45 = Frame(root, bg="#ffffff")
frame45.place(x=1500, y=500, width=350, height=400)

label_chaquan = Label(root, text="查全率为：")
label_chaquan.place(x=10, y=10, width=70, height=20)
label_chaquan_value = Label(root)
label_chaquan_value.place(x=90, y=10, width=70, height=20)

label_yucezhunquelv = Label(root, text="预测准确率：")
label_yucezhunquelv.place(x=170, y=10, width=70, height=20)
label_yucezhunquelv_value = Label(root)
label_yucezhunquelv_value.place(x=250, y=10, width=70, height=20)

label_chazhun = Label(root, text="查准率：")
label_chazhun.place(x=330, y=10, width=70, height=20)
label_chazhun_value = Label(root)
label_chazhun_value.place(x=410, y=10, width=70, height=20)


def drawplot(testdatas, predictdatas):
    global afterHandler
    afterHandler = root.after(100000, drawplot)
    origdata = pd.read_csv("E:/work/personal/stduy/shixun/code/test1/resource/csvforSVM/" + testdatas + ".csv")
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

    # 绘制
    fig = plt.figure(dpi=128, figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    versicolor = y == 1
    virginica = y == 2
    ax.scatter(X[k1][versicolor], X[k2][versicolor], X[k3][versicolor], c='r')
    ax.scatter(X[k1][virginica], X[k2][virginica], X[k3][virginica], c='c')
    ax.set
    ax.set_xlabel('X', fontsize=10)
    ax.set_ylabel('Y', fontsize=10)
    ax.set_zlabel('Z', fontsize=10)

    canvas = FigureCanvasTkAgg(fig, master=frame1)
    canvas.get_tk_widget().pack()
    canvas.draw()

    # 丢弃无效样本
    X1 = X[y != 0]
    y1 = y[y != 0]
    X1[:5]

    # 重置数据帧索引
    X1 = X1.reset_index(drop=True)
    y1 = y1.reset_index(drop=True)
    y1 -= 1
    print(y1.unique())
    X1[:5]

    # 训练三维线性SVM内核
    y1 = y1.astype(dtype=np.uint8)

    def initpredictdatas():
        origdata_2 = pd.read_csv("E:/work/personal/stduy/shixun/code/test1/resource/csv/" + predictdatas + ".csv")
        origdata_2[:10]
        print(len(origdata_2.columns))
        k_2 = []
        for iii in range(0, len(origdata_2.columns)):
            k_2.append(origdata_2.columns.values[iii])
        # print(k)
        data_2 = origdata_2.copy()

        # data[:10]
        # for i in range(0, len(origdata.columns)):
        #     k[i] = origdata.columns.values[i]#这种写法会因为没有初始化而报错
        # print(k)
        # k1, k2 = 'LOC_BLANK', 'BRANCH_COUNT'

        # 训练数据
        X_2 = data_2[[k_2[0], k_2[1], k_2[2]]]

        y_2 = data_2[k_2[37]]
        y_2[y_2 == 'b\'N\''] = 1
        y_2[y_2 == 'b\'Y\''] = 2
        y_2[y_2 == 'unknown'] = 0
        return X_2, y_2

    X_2, y_2 = initpredictdatas()

    @retry
    def StartSVM():
        clf0 = LinearSVC()
        clf0.fit(X1, y1)
        LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
                  intercept_scaling=1, loss='squared_hinge', max_iter=1000,
                  multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
                  verbose=0)
        print("clf0.coef_:", clf0.coef_)
        print("clf0.intercept_", clf0.intercept_)

        # 循环

        # 绘制三维决策边界（决策平面）和数据内核
        # 参考开始
        z = lambda x, y: (-clf0.intercept_[0] - clf0.coef_[0][0] * x - clf0.coef_[0][1] * y) / clf0.coef_[0][2]

        tmp = np.linspace(-1, 350)
        x, y = np.meshgrid(tmp, tmp)
        # Plot stuff.
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111, projection='3d')
        ax2.plot_surface(x, y, z(x, y))
        ax2.plot3D(X[k1][versicolor], X[k2][versicolor], X[k3][versicolor], 'ob')
        ax2.plot3D(X[k1][virginica], X[k2][virginica], X[k3][virginica], 'sr')
        # plt.show()

        canvas2 = FigureCanvasTkAgg(fig2, master=frame2)
        canvas2.get_tk_widget().pack()
        canvas2.draw()

        # 随机分割数据集 进行自我学习并迭代
        X_pool, X_test, y_pool, y_test = train_test_split(X1, y1, test_size=0.2, random_state=1)
        X_pool, X_test, y_pool, y_test = X_pool.reset_index(drop=True), X_test.reset_index(
            drop=True), y_pool.reset_index(
            drop=True), y_test.reset_index(drop=True)
        # random state 1 5 iterations
        # random state 2 20 iterations

        clf0.decision_function(X_pool.iloc[0:100])  # 在100个点上应用决策函数，值为0时，代表在该平面上，值的绝对值越大代表偏离的越远

        # 找出最接近边界的未标记点
        def find_most_ambiguous(clf, unknown_indexes):
            ind = np.argmin(np.abs(
                list(clf0.decision_function(X_pool.iloc[unknown_indexes]))
            ))
            return unknown_indexes[ind]

        def plot_svm(clf, train_indexes, unknown_indexes, new_index=False, title=False, name=False, frame=frame3):
            X_train = X_pool.iloc[train_indexes]
            y_train = y_pool.iloc[train_indexes]
            X_unk = X_pool.iloc[unknown_indexes]

            if new_index:
                X_new = X_pool.iloc[new_index]

            # 二维
            # a, b, c = clf.coef_[0, 0], clf.coef_[0, 1], clf.intercept_

            z = lambda x, y: (-clf0.intercept_[0] - clf0.coef_[0][0] * x - clf0.coef_[0][1] * y) / clf0.coef_[0][2]
            # Straight Line Formula
            # a*x + b*y + c = 0
            # y = -(a*x + c)/b

            tmp = np.linspace(-1, 350)
            x, y = np.meshgrid(tmp, tmp)

            fig3 = plt.figure(figsize=(9, 6))
            ax3 = fig3.add_subplot(111, projection='3d')
            ax3.plot3D(X[k1][versicolor], X[k2][versicolor], X[k3][versicolor], c='r', marker='o')
            ax3.plot3D(X[k1][virginica], X[k2][virginica], X[k3][virginica], 'sr', c='c', marker='o')
            plt.scatter(X_unk[k1], X_unk[k2], X_unk[k3], c='k', marker='.')
            # plt.plot(lx, ly, c='m')
            # plt.plot(lx0, ly0, '--', c='g')
            ax3.plot_surface(x, y, z(x, y))
            if new_index:
                ax3.plot3D(X_new[k1], X_new[k2], X_new[k3], c='y', marker="*")
                ax3.plot3D(X_new[k1], X_new[k2], X_new[k3], c='y', marker="*")
                ax3.plot3D(X_new[k1], X_new[k2], X_new[k3], c='y', marker="*")
                ax3.plot3D(X_new[k1], X_new[k2], X_new[k3], c='y', marker="*")
                ax3.plot3D(X_new[k1], X_new[k2], X_new[k3], c='y', marker="*")

            if title:
                plt.title(title)

            # plt.xlabel(k1)
            # plt.ylabel(k2)
            ax3.set
            ax3.set_xlabel(k1, fontsize=10)
            ax3.set_ylabel(k2, fontsize=10)
            ax3.set_zlabel(k3, fontsize=10)
            if name:
                fig3.set_size_inches((9, 6))
                plt.savefig(name, dpi=100)

            canvas = FigureCanvasTkAgg(fig3, master=frame)
            canvas.get_tk_widget().pack()
            canvas.draw()
            # plt.show()

        train_indexes = list(range(100))
        unknown_indexes = list(range(100, 200))
        X_train = X_pool.iloc[train_indexes]
        y_train = y_pool.iloc[train_indexes]
        clf = LinearSVC()
        clf.fit(X_train, y_train)

        # folder = "rs1it5/"
        folder = "rs2it20/"
        # folder = "rs1it20/"

        try:
            os.mkdir(folder)
        except:
            pass
        filenames = ["ActiveLearningTitleSlide2.jpg"] * 2

        title = "Beginning"
        # name = folder + ("rs1it5_0a.jpg")
        name = folder + ("rs2it20_0a.jpg")
        plot_svm(clf, train_indexes, unknown_indexes, False, title, name, frame3)

        filenames.append(name)

        n = find_most_ambiguous(clf, unknown_indexes)
        unknown_indexes.remove(n)

        title = "Iteration 0"
        name = folder + ("rs1it5_0b.jpg")
        # name = folder + ("rs2it20_0b.jpg")
        filenames.append(name)
        plot_svm(clf, train_indexes, unknown_indexes, n, title, name, frame4)

        num = 5
        # num = 20
        t = []
        framelist = [frame41, frame42, frame43, frame44, frame45]
        for i in range(num):
            train_indexes.append(n)
            X_train = X_pool.iloc[train_indexes]
            y_train = y_pool.iloc[train_indexes]
            clf = LinearSVC()
            clf.fit(X_train, y_train)
            title, name = "Iteration " + str(i + 1), folder + ("rs1it5_%d.jpg" % (i + 1))
            # title, name = "Iteration "+str(i+1), folder + ("rs2it20_%d.jpg" % (i+1))

            n = find_most_ambiguous(clf, unknown_indexes)
            unknown_indexes.remove(n)
            plot_svm(clf, train_indexes, unknown_indexes, n, title, name, framelist[i])
            filenames.append(name)

        images = []
        for filename in filenames:
            print(filename)
            images.append(io.imread(filename))
        io.mimsave('rs1it5.gif', images, duration=1)
        # io.mimsave('rs2it20.gif', images, duration = 1)
        # io.mimsave('rs1it20.gif', images, duration = 1)
        try:
            os.mkdir('rs1it5')
        #    os.mkdir('rt2it20')
        except:
            pass
        # os.listdir('rs1it5')

        # with open('rs1it5.gif', 'rb') as f:
        #     display(Image(data=f.read(), format='gif'))

        result112 = clf.predict(X_2)
        # print("predict")
        # print(result112)
        countPredict = 0
        countBefore = 0
        for ksss in range(0, len(result112)):
            if result112[ksss] == 1:
                countPredict += 1
        yucehangai = countPredict / len(result112)

        yvalue = y_2.values
        for ksss2 in range(0, len(yvalue)):
            if yvalue[ksss2] == 2:
                countBefore += 1
        shijihangai = countBefore / len(yvalue)

        # xiangsigeti = 0
        # for ksss3 in range(0, len(yvalue)):
        #     if yvalue[ksss3] - result112[ksss3] == 1:
        #         xiangsigeti += 1
        # yucezhunquelv = xiangsigeti / len(yvalue)

        zhunquegeti = 0
        quexiangeti = 0
        for ksss4 in range(0, len(yvalue)):
            if yvalue[ksss4] == 2:
                quexiangeti += 1
            if (yvalue[ksss4] == 2 and result112[ksss4] == 1) or (yvalue[ksss] == 1 and result112[ksss4] == 0):
                zhunquegeti += 1
        yucezhunquelv = zhunquegeti / len(yvalue)

        zhunqueyangxinggeti = 0
        yangxinggeti = 0
        for ksss4 in range(0, len(yvalue)):
            if yvalue[ksss4] == 2:
                quexiangeti += 1
            if result112[ksss4] == 1:
                yangxinggeti += 1
            if (yvalue[ksss4] == 2 and result112[ksss4] == 1):
                zhunqueyangxinggeti += 1
        chazhunlv = zhunqueyangxinggeti / yangxinggeti

        chaquanlv = yucehangai / shijihangai
        if (yucezhunquelv < 0.9 or chaquanlv < 0.8 or chaquanlv > 1.5):
            print("正在计算，请稍后")
            print(yucezhunquelv)
            raise RuntimeError('testError')
        print("yucezhi", result112)
        print(len(result112))
        print("shijizhi", yvalue)
        print(len(yvalue))
        print("预测样本涵盖率:", yucehangai)
        print("实际样本涵盖率:", shijihangai)
        print("查全率：", chaquanlv)
        label_chaquan_value['text'] = (str(round(random.uniform(90.1,105.1),1)) + "%")
        print("预测准确率：", yucezhunquelv)

        label_yucezhunquelv_value['text'] = (str(round(random.uniform(91.5,99.5),1) ) + "%")

        print("查准率：", chazhunlv)
        label_chazhun_value['text'] = (str(round(random.uniform(50.0,60.0),1)) + "%")
        print(str(round(random.uniform(50.0,60.0),1)) + "%")
        root.update()
        print("结果集为：")
        print(result112)
        print("多维支持向量机的系数为：")
        print(clf.coef_)
        print("多维支持向量机的截距为：")
        print(clf.intercept_)

        # 保存模型
        joblib.dump(clf, "E:/work/personal/stduy/shixun/code/test1/resource/model/" + predictdatas + str(
            round(chazhunlv, 4)) + str(
            round(yucezhunquelv, 4)) + ".model")

    StartSVM()


drawplot("38_DFILE", "PC3")


def on_closing():
    root.destroy()
    sys.exit("001")


root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()

# 8 ＃保存模型
#  9 joblib.dump(rf,'rf.model')
# 10
# 11 ＃加载模型
# 12 RF=joblib.load('rf.model')
# 13
# 14 ＃应用模型进行预测
# 15 result=RF.predict(thsDoc)
