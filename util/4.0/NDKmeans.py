from scipy.io import arff
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from retrying import retry
import time

column_number = 0


def startstart(filename0):
    # arff to txt
    # 1.把arff转成csv以便后期拿来预测
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

    # csv转无列标题的txt
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
                        f.write(str(line[i]) + ',')
                    else:
                        # print(line[i],"      ",i)
                        f.write(str(line[i]) + '\n')

    start(filename0)
    start2(filename0)


# 使用txt文件进行K-MEANS
@retry
def start(filename_kmeans):
    # 加载数据
    def loadDataSet(fileName):
        data = np.loadtxt("resource/txt/" + fileName + ".txt", delimiter=',')
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


def chulitxtwenjianzhunbei(filenameforall):
    def updateFile(file, old_str, new_str):
        """
        替换文件中的字符串
        :param file:文件名
        :param old_str:旧字符串
        :param new_str:新字符串
        :return:
        """
        file_data = ""
        with open(file, "r") as f:
            for line in f:
                line = line.replace(old_str, new_str)
                file_data += line
        with open(file, "w") as f:
            f.write(file_data)

    updateFile(filenameforall, ' ', ',')
    df_example_noCols = pd.read_csv("PC1result.txt", header=None,
                                    names=["LOC_BLANK", "BRANCH_COUNT", "CALL_PAIRS", "LOC_CODE_AND_COMMENT",
                                           "LOC_COMMENTS", "CONDITION_COUNT", "CYCLOMATIC_COMPLEXITY",
                                           "CYCLOMATIC_DENSITY",
                                           "DECISION_COUNT", "DECISION_DENSITY", "DESIGN_COMPLEXITY",
                                           "DESIGN_DENSITY", "EDGE_COUNT", "ESSENTIAL_COMPLEXITY", "ESSENTIAL_DENSITY",
                                           "LOC_EXECUTABLE", "PARAMETER_COUNT", "HALSTEAD_CONTENT",
                                           "HALSTEAD_DIFFICULTY",
                                           "HALSTEAD_EFFORT", "HALSTEAD_ERROR_EST",
                                           "HALSTEAD_LENGTH", "HALSTEAD_LEVEL", "HALSTEAD_PROG_TIME", "HALSTEAD_VOLUME",
                                           "MAINTENANCE_SEVERITY", "MODIFIED_CONDITION_COUNT",
                                           "MULTIPLE_CONDITION_COUNT",
                                           "NODE_COUNT", "NORMALIZED_CYLOMATIC_COMPLEXITY",
                                           "NUM_OPERANDS", "NUM_OPERATORS", "NUM_UNIQUE_OPERANDS",
                                           "NUM_UNIQUE_OPERATORS",
                                           "NUMBER_OF_LINES", "PERCENT_COMMENTS", "LOC_TOTAL", "Defective"])
    dfc = df_example_noCols
    dfc.to_csv('dfc.csv', index=None)


filePath = '/resource'
filenames=os.listdir(filePath)
for i in range(0,len(filenames)):
    1==1

filenameforall = "PC5"
startstart(filenameforall)

time.sleep(5)

start(filenameforall)

filenameforall2=''
chulitxtwenjianzhunbei()
