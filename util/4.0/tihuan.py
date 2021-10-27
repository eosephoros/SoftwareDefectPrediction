import pandas as pd

data = pd.read_csv("PC1result.csv")
gender = {1.0: 'b\'N\'', 0.0: 'b\'Y\''}
data["Defective"] = data["Defective"].map(gender)
data.to_csv('PC1result1.csv')
print(data)
#      用户编号    性别   年龄(岁)   年收入(元)  是否购买
# 0   15624510      1     19         19000         0
# 1   15810944      1     35         20000         0
# 2   15668575      2     26         43000         0
# 3   15603246      2     27         57000         0
