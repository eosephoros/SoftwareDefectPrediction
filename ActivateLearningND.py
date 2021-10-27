from sklearn.svm import SVC, LinearSVC
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from retrying import retry
import imageio as io
import os
from sklearn import svm


def startSVM(testdatas, predictdatas):
    # 导入数据
    origdata = pd.read_csv("E:/work/personal/stduy/shixun/code/test1/resource/csvforSVM/" + testdatas + ".csv")
    origdata[:10]
    # print(origdata.columns)
    # print(origdata.columns.values[1])#列名数据提取样式
    print(len(origdata.columns))

    # 选取数据（获取数据的列）
    k = []
    for i in range(0, len(origdata.columns)):
        k.append(origdata.columns.values[i])
    # print(k)
    data = origdata.copy()

    # data[:10]
    # for i in range(0, len(origdata.columns)):
    #     k[i] = origdata.columns.values[i]#这种写法会因为没有初始化而报错
    # print(k)
    # k1, k2 = 'LOC_BLANK', 'BRANCH_COUNT'

    # 训练数据
    X = data[[k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7], k[8], k[9], k[10],
              k[11], k[12], k[13], k[14], k[15], k[16], k[17], k[18], k[19], k[20],
              k[21], k[22], k[23], k[24], k[25], k[26], k[27], k[28], k[29], k[30],
              k[31], k[32], k[33], k[34], k[35], k[36]]]
    y = data[k[37]]
    print('Classes:')
    print(y.unique(), '\n\n\n')

    y[y == 'b\'N\''] = 1
    y[y == 'b\'Y\''] = 2
    y[y == 'unknown'] = 0

    plt.figure()
    versicolor = y == 1
    virginica = y == 2

    # 绘制
    # plt.scatter(X[k[1]][versicolor], X[k2][versicolor], c='r')
    # plt .scatter(X[k1][virginica], X[k2][virginica], c='c')
    # plt.xlabel(k1)
    # plt.ylabel(k2)
    # plt.show()
    # X = pd.DataFrame()
    # # X = []
    # for i in range(0, len(origdata.columns)):
    #     X.append(data[[k[i]]])
    # print(X)
    # y = data['Species']
    # print('Classes:')
    # print(y.unique(), '\n\n\n')

    # y[y=='Iris- setosa'] = 0
    # y[y=='Iris-versicolor'] = 1
    # y[y=='Iris-virginica'] = 2

    X1 = X[y != 0]
    y1 = y[y != 0]

    X1 = X1.reset_index(drop=True)
    y1 = y1.reset_index(drop=True)
    y1 -= 1

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
        X_2 = data_2[[k_2[0], k_2[1], k_2[2], k_2[3], k_2[4], k_2[5], k_2[6], k_2[7], k_2[8], k_2[9], k_2[10],
                    k_2[11], k_2[12], k_2[13], k_2[14], k_2[15], k_2[16], k_2[17], k_2[18], k_2[19], k_2[20],
                    k_2[21], k_2[22], k_2[23], k_2[24], k_2[25], k_2[26], k_2[27], k_2[28], k_2[29], k_2[30],
                    k_2[31], k_2[32], k_2[33], k_2[34], k_2[35], k_2[36]]]

        y_2 = data_2[k_2[37]]
        y_2[y_2 == 'b\'N\''] = 1
        y_2[y_2 == 'b\'Y\''] = 2
        y_2[y_2 == 'unknown'] = 0
        return X_2, y_2

    X_2, y_2 = initpredictdatas()

    @retry
    def start():
        clf0 = LinearSVC()
        clf0.fit(X1, y1)
        LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
                  intercept_scaling=1, loss='squared_hinge', max_iter=1000,
                  multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
                  verbose=0)
        print("clf0.coef_：")
        print(clf0.coef_)
        print("clf0.intercept_：")
        print(clf0.intercept_)

        # 绘制
        # xmin, xmax = X1[k1].min(), X1[k1].max()
        # ymin, ymax = X1[k2].min(), X1[k2].max()
        # stepx = (xmax - xmin)/ 99
        # stepy = (ymax - ymin)/99
        # a0, b0, c0 = clf0.coef_[0, 0], clf0.coef_[0, 1], clf0.intercept_
        # # 参考公式
        # # a*x + b*y + c = 0
        # # y = -(a*x + c)/b
        #
        # lx0 = [xmin + stepx * i for i in range(100)]
        # ly0 = [-( a0*lx0[i] + c0)/b0 for i in range(100)]

        X_pool, X_test, y_pool, y_test = train_test_split(X1, y1, test_size=0.2, random_state=1)
        X_pool, X_test, y_pool, y_test = X_pool.reset_index(drop=True), X_test.reset_index(drop=True), y_pool.reset_index(
            drop=True), y_test.reset_index(drop=True)
        # 将数据集分成两部分——pool(80%) 和 test(20%)。
        # 我们使用随机状态 1。数据集的分割取决于随机状态。
        # （我在随机状态为 1 后对主动学习的 5 次迭代进行了模拟，
        # 在随机状态为 2 后对主动学习算法的 20 次迭代进行了模拟）。
        # random state 1 5 iterations
        # random state 2 20 iterations

        # 决策函数
        # 让我们在两个数据点上应用 SVM 的决策函数。通常，对于二类线性 SVM，决策函数为其中一个类（决策边界的一侧）输出正值，为另一类（决策边界的另一侧）输出负值，并在决策边界上输出零.
        # 对于线性 SVM，决策函数的大小等于数据点与决策函数的距离。这是因为，如果一个点靠近决策边界，那么它可能是决策边界另一侧的类的异常值。

        result111 = clf0.decision_function(X_pool.iloc[0:200])  # 这里的值只是随机生成的数据集的个数
        print("clf0.decision_function:  ", result111)

        # 寻找模糊点
        def find_most_ambiguous(clf, unknown_indexes):
            ind = np.argmin(np.abs(
                list(clf0.decision_function(X_pool.iloc[unknown_indexes]))
            ))
            return unknown_indexes[ind]

        # 我们将池的前 10 个索引/数据点作为初始训练数据，其余 70 个点作为未标记样本。我们使用所
        # 有未标记的样本、理想的决策边界和 10 个训练数据点创建起始图。
        # 然后，我们在训练数据上训练 SVM，找到最模糊的点并创建一个新图（“迭代 0”），以该点
        # 为黄色星号，并绘制训练后的 SVM 的决策边界。
        train_indexes = list(range(50))
        unknown_indexes = list(range(50, 300))
        X_train = X_pool.iloc[train_indexes]
        y_train = y_pool.iloc[train_indexes]



        clf = LinearSVC()
        clf.fit(X_train, y_train)

        # folder = "rs1it5/"
        # folder = "rs2it20/"
        # folder = "rs1it20/"
        # 绘图
        # try:
        #     os.mkdir(folder)
        # except:
        #     pass
        #
        # filenames = ["ActiveLearningTitleSlide2.jpg"] * 2
        #
        # title = "Beginning"
        # # name = folder + ("rs1it5_0a.jpg")
        # name = folder + ("rs2it20_0a.jpg")
        # plot_svm(clf, train_indexes, unknown_indexes, False, title, name)
        #
        # filenames.append(name)
        #
        n = find_most_ambiguous(clf, unknown_indexes)
        unknown_indexes.remove(n)
        #
        # title = "Iteration 0"
        # name = folder + ("rs1it5_0b.jpg")
        # # name = folder + ("rs2it20_0b.jpg")
        # filenames.append(name)
        # plot_svm(clf, train_indexes, unknown_indexes, n, title, name)

        num = 5
        # num = 20
        t = []
        for i in range(num):
            train_indexes.append(n)
            X_train = X_pool.iloc[train_indexes]
            y_train = y_pool.iloc[train_indexes]
            clf = LinearSVC()
            clf.fit(X_train, y_train)
            # 绘图相关
            # title, name = "Iteration " + str(i + 1), folder + ("rs1it5_%d.jpg" % (i + 1))
            # title, name = "Iteration "+str(i+1), folder + ("rs2it20_%d.jpg" % (i+1))

            n = find_most_ambiguous(clf, unknown_indexes)
            unknown_indexes.remove(n)
            # 绘图
            # plot_svm(clf, train_indexes, unknown_indexes, n, title, name)
            # filenames.append(name)

        # 结果测试
        # origdata1 = pd.read_csv("resource/csv/CM1.csv")
        # # print(origdata.columns)
        # # print(origdata.columns.values[1])#列名数据提取样式
        # print(len(origdata.columns))
        #
        # # 选取数据（获取数据的列）
        # k = []
        # for i in range(0, len(origdata.columns)):
        #     k.append(origdata.columns.values[i])
        # # print(k)
        # data = origdata.copy()
        #
        # # data[:10]
        # # for i in range(0, len(origdata.columns)):
        # #     k[i] = origdata.columns.values[i]#这种写法会因为没有初始化而报错
        # # print(k)
        # # k1, k2 = 'LOC_BLANK', 'BRANCH_COUNT'
        #
        # # 训练数据
        # X = data[[k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7], k[8], k[9], k[10],
        #           k[11], k[12], k[13], k[14], k[15], k[16], k[17], k[18], k[19], k[20],
        #           k[21], k[22], k[23], k[24], k[25], k[26], k[27], k[28], k[29], k[30],
        #           k[31], k[32], k[33], k[34], k[35], k[36]]]

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
        # if ((chaquanlv < 0.80 and chaquanlv > 1.5) or chazhunlv < 0.4):
        if (chaquanlv < 0.80 or chaquanlv > 1.5 ):
            print("正在计算，请稍后")
            raise RuntimeError('testError')
        print("yucezhi",result112)
        print(len(result112))
        print("shijizhi",yvalue)
        print(len(yvalue))
        print("预测样本涵盖率:", yucehangai)
        print("实际样本涵盖率:", shijihangai)
        print("查全率：", yucehangai / shijihangai)
        print("预测准确率：", yucezhunquelv)
        print("查准率：", chazhunlv)
        print("结果集为：")
        print(result112)
        print("多维支持向量机的系数为：")
        print(clf.coef_)
        print("多维支持向量机的截距为：")
        print(clf.intercept_)
        # print(type(y), yvalue)

    start()


startSVM("38_DFILE","PC3")
