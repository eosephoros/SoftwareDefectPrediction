from scipy.io import arff
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from retrying import retry
import time

column_number = 0


def startstart(filename0):
    def start(filename):
        # arff to csv
        file_name = 'E:/work/personal/stduy/shixun/code/test1/resource/arff/' + filename + '.arff'

        data, meta = arff.loadarff(file_name)
        # print(data)
        # print(meta)

        df = pd.DataFrame(data)
        print(df.head())

        ##读取列数
        columns = df.columns
        global column_number
        column_number = len(columns)
        print(len(columns))
        # print(df)
        # 先删除再添加
        if os.path.exists('resource/csv/' + filename + '.csv'):
            os.remove('resource/csv/' + filename + '.csv')
        else:
            print("The file does not exist")
        # 保存为csv文件
        out_file = 'E:/work/personal/stduy/shixun/code/test1/resource/csv/' + filename + '.csv'
        output = pd.DataFrame(df)
        output.to_csv(out_file, index=False)

    def start2(filename):
        data = pd.read_csv('resource/csv/' + filename + '.csv', encoding='utf-8')
        housing_map = {"b'N'": 1, "b'Y'": 0}
        try:
            data['label'] = data['label'].map(housing_map)
        except:
            data['Defective'] = data['Defective'].map(housing_map)
        # 先删除再添加
        if os.path.exists('resource/txt/' + filename + '.txt'):
            os.remove('resource/txt/' + filename + '.txt')
        else:
            print("The file does not exist")
        with open('resource/txt/' + filename + '.txt', 'a+', encoding='utf-8') as f:
            for line in data.values:
                for i in range(0, column_number):
                    if i < column_number - 1:
                        f.write(str(line[i]) + '\t')
                    else:
                        # print(line[i],"      ",i)
                        f.write(str(line[i]) + '\n')
                    # f.write((str(line[0]) + '\t' + str(line[1]) + '\t' + str(line[2]) + '\t' + str(line[3]) + '\t' + str(
                    #     line[4]) + '\t' + str(line[5]) + '\t' +
                    #      str(line[6]) + '\t' + str(line[7]) + '\t' + str(line[8]) + '\t' + str(line[9]) + '\t' + str(
                    #         line[10]) + '\t' + str(line[11]) + '\t' + str(line[12]) + '\t' + str(line[13])
                    #      + '\t' + str(line[14]) + '\t' + str(line[15]) + '\t' + str(line[16]) + '\t' + str(
                    #         line[17]) + '\t' + str(line[18]) + '\t' + str(line[19]) + '\t'
                    #      + str(line[20]) + '\t' + str(line[21]) + '\t' + str(line[22]) + '\t' + str(line[23]) + '\t'
                    #      + str(line[24]) + '\t' + str(line[25]) + '\t' + str(line[26]) + '\t' + str(line[27]) + '\t'
                    #      + str(line[28]) + '\t' + str(line[29]) + '\t' + str(line[30]) + '\t' + str(line[31]) + '\t'
                    #      + str(line[32]) + '\t' + str(line[33]) + '\t' + str(line[34]) + '\t' + str(line[35]) + '\t'
                    #      + str(line[36]) + '\t' + str(line[37]) + '\n'))

    start(filename0)
    start2(filename0)







@retry
def start(filename_kmeans):
    # 加载数据
    def loadDataSet(fileName):
        data = np.loadtxt("resource/txt/" + fileName + ".txt", delimiter='\t')
        # data = np.loadtxt(fileName, delimiter='\t', dtype=float, skiprows=1)
        return data
        # data =  np.loadtxt(fileName)

    # 欧氏距离计算
    def distEclud(x, y):
        return np.sqrt(np.sum((x - y) ** 2))

    # 为给定数据集构建一个包含K个随机质心的集合
    def randCent(dataSet, k):
        # 获取样本数与特征值
        m, n = dataSet.shape  # 把数据集的行数和列数赋值给m,n
        # 初始化质心,创建(k,n)个以零填充的矩阵
        centroids = np.zeros((k, n))
        # 循环遍历特征值
        for i in range(k):
            index = int(np.random.uniform(0, m))
            # 计算每一列的质心,并将值赋给centroids
            centroids[i, :] = dataSet[index, :]
            # 返回质心
        return centroids

    # k均值聚类
    def KMeans(dataSet, k):
        m = np.shape(dataSet)[0]
        # 初始化一个矩阵来存储每个点的簇分配结果
        # clusterAssment包含两个列:一列记录簇索引值,第二列存储误差(误差是指当前点到簇质心的距离,后面会使用该误差来评价聚类的效果)
        clusterAssment = np.mat(np.zeros((m, 2)))
        clusterChange = True
        cloestPoint = clusterAssment

        # 创建质心,随机K个质心
        centroids = randCent(dataSet, k)
        # 初始化标志变量,用于判断迭代是否继续,如果True,则继续迭代
        while clusterChange:
            clusterChange = False

            # 遍历所有样本（行数）
            for i in range(m):
                minDist = 100000.0
                minIndex = -1
                # 遍历所有数据找到距离每个点最近的质心,
                # 可以通过对每个点遍历所有质心并计算点到每个质心的距离来完成
                for j in range(k):
                    # 计算数据点到质心的距离
                    # 计算距离是使用distMeas参数给出的距离公式,默认距离函数是distEclud
                    distance = distEclud(centroids[j, :], dataSet[i, :])
                    # 如果距离比minDist(最小距离)还小,更新minDist(最小距离)和最小质心的index(索引)
                    if distance < minDist:
                        minDist = distance
                        minIndex = j
                # 如果任一点的簇分配结果发生改变,则更新clusterChanged标志
                if clusterAssment[i, 0] != minIndex:
                    clusterChange = True
                    # 更新簇分配结果为最小质心的index(索引),minDist(最小距离)
                    clusterAssment[i, :] = minIndex, minDist
            # 遍历所有质心并更新它们的取值
            for j in range(k):
                # 通过数据过滤来获得给定簇的所有点
                pointsInCluster = dataSet[np.nonzero(clusterAssment[:, 0].A == j)[0]]
                # 计算所有点的均值,axis=0表示沿矩阵的列方向进行均值计算
                centroids[j, :] = np.mean(pointsInCluster, axis=0)
        print("Congratulation,cluster complete!")
        # 返回所有的类质心与点分配结果
        return centroids, clusterAssment, pointsInCluster

    def showCluster(dataSet, k, centroids, clusterAssment, filename):
        m, n = dataSet.shape
        # if n != 2:
        # print("数据不是二维的")
        # return 1

        # mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
        # if k > len(mark):
        #     print("k值太大了")
        #     return 1
        # 绘制所有样本
        for i in range(m):
            markIndex = int(clusterAssment[i, 0])
            plt.plot(dataSet[i, 0], dataSet[i, 1], marker='.', color='black', markersize=4)

        mark2 = ['*', '.', 'o', '|', '^b', '+b', 'sb', 'db', '<b', 'pb']
        # 绘制质心
        for i in range(k):
            plt.plot(centroids[i, 0], centroids[i, 1], marker='*', color='red', markersize=20)

        # 控制数量
        print(len(pointsInCluster))
        if (len(pointsInCluster) < 15):
            raise RuntimeError('testError')
        # 先删除再添加
        if os.path.exists('resource/result/' + filename + "result" + '.txt'):
            os.remove('resource/result/' + filename + "result" + '.txt')
        else:
            print("The file does not exist")
        np.savetxt('resource/result/' + filename + "result" + ".txt", pointsInCluster, fmt='%1.2f')

    filename = filename_kmeans
    dataSet = loadDataSet(filename)
    k = 20
    centroids, clusterAssment, pointsInCluster = KMeans(dataSet, k)
    print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
    print(pointsInCluster)
    print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
    showCluster(dataSet, k, centroids, clusterAssment, filename)


filenameforall="PC5"
startstart(filenameforall)


time.sleep(5)

start(filenameforall)
