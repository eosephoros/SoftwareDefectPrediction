import os

import pandas as pd



def start(filename):
    data = pd.read_csv(filename + '.csv', encoding='utf-8')
    housing_map = {"b'N'": 1, "b'Y'": 0}
    data['Defective'] = data['Defective'].map(housing_map)
    # 先删除再添加
    if os.path.exists('noCT' + filename + '.txt'):
        os.remove('noCT' + filename + '.txt')
    else:
        print("The file does not exist")
    with open('noCT' + filename + '.txt', 'a+', encoding='utf-8') as f:
        for line in data.values:
            for i in range(0, 38):
                if i < 37:
                    f.write(str(line[i]) + ',')
                else:
                    print(line[i],"      ",i)
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
start("CM1")