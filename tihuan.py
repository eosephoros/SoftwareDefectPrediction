import os
import pandas as pd


def tihuan(filename):
    data = pd.read_csv("E:/work/personal/stduy/shixun/code/test1/resource/KMeansResultCSV/"+filename+".csv")
    gender = {1.0: 'b\'N\'', 0.0: 'b\'Y\''}

    # try:
    #     data['label'] = data['label'].map(housing_map)
    # except:
    #     data['Defective'] = data['Defective'].map(housing_map)
    try:
        data["Defective"] = data["Defective"].map(gender)
    except:
        data["label"] = data["label"].map(gender)
    data.to_csv('E:/work/personal/stduy/shixun/code/test1/resource/csvforSVM/'+filename+'ForSVM.csv',index=None)
    print(data)


os.chdir("E:\\work\\personal\\stduy\\shixun\code\\test1\\resource\\KMeansResultCSV")
filenames = os.listdir(".")
names = []
for i in filenames:
    portion = os.path.splitext(i)  # 把文件名拆分为名字和后缀
    if portion[1] == ".arff":
        names.append(portion[0])
    if portion[1] == ".csv":
        names.append(portion[0])
    if portion[1] == ".txt":
        names.append(portion[0])
for i in range(0, len(names)):
    print(names[i])
    tihuan(names[i])
