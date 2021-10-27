import os
import pandas as pd


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


updateFile("PC1result.txt", ' ', ',')

origdata = pd.read_csv("../../resource/csv/CM1.csv")
origdata[:10]
print(len(origdata.columns))
# # 选取数据（获取数据的列）
k = []
for i in range(0, len(origdata.columns)):
    k.append(origdata.columns.values[i])  # print(origdata.columns.values[1])#列名数据提取样式
# # 读数后加标题
# df_example_noCols = pd.read_csv("PC1result.txt", header=None,
#                                 names=["LOC_BLANK", "BRANCH_COUNT", "CALL_PAIRS", "LOC_CODE_AND_COMMENT",
#                                        "LOC_COMMENTS", "CONDITION_COUNT", "CYCLOMATIC_COMPLEXITY", "CYCLOMATIC_DENSITY",
#                                        "DECISION_COUNT", "DECISION_DENSITY", "DESIGN_COMPLEXITY",
#                                        "DESIGN_DENSITY", "EDGE_COUNT", "ESSENTIAL_COMPLEXITY", "ESSENTIAL_DENSITY",
#                                        "LOC_EXECUTABLE", "PARAMETER_COUNT", "HALSTEAD_CONTENT", "HALSTEAD_DIFFICULTY",
#                                        "HALSTEAD_EFFORT", "HALSTEAD_ERROR_EST",
#                                        "HALSTEAD_LENGTH", "HALSTEAD_LEVEL", "HALSTEAD_PROG_TIME", "HALSTEAD_VOLUME",
#                                        "MAINTENANCE_SEVERITY", "MODIFIED_CONDITION_COUNT", "MULTIPLE_CONDITION_COUNT",
#                                        "NODE_COUNT", "NORMALIZED_CYLOMATIC_COMPLEXITY",
#                                        "NUM_OPERANDS", "NUM_OPERATORS", "NUM_UNIQUE_OPERANDS", "NUM_UNIQUE_OPERATORS",
#                                        "NUMBER_OF_LINES", "PERCENT_COMMENTS", "LOC_TOTAL", "Defective"])
df_example_noCols = pd.read_csv("PC1result.txt", header=None)
df_example_noCols.columns=k
df1=df_example_noCols
df1.to_csv("PC1result.csv",index=None)
print(df_example_noCols.columns)
# 最后保存到dfc
# df_example_noCols.columns = ["LOC_BLANK", "BRANCH_COUNT", "CALL_PAIRS", "LOC_CODE_AND_COMMENT",
#                              "LOC_COMMENTS", "CONDITION_COUNT", "CYCLOMATIC_COMPLEXITY", "CYCLOMATIC_DENSITY",
#                              "DECISION_COUNT", "DECISION_DENSITY", "DESIGN_COMPLEXITY",
#                              "DESIGN_DENSITY", "EDGE_COUNT", "ESSENTIAL_COMPLEXITY", "ESSENTIAL_DENSITY",
#                              "LOC_EXECUTABLE", "PARAMETER_COUNT", "HALSTEAD_CONTENT", "HALSTEAD_DIFFICULTY",
#                              "HALSTEAD_EFFORT", "HALSTEAD_ERROR_EST",
#                              "HALSTEAD_LENGTH", "HALSTEAD_LEVEL", "HALSTEAD_PROG_TIME", "HALSTEAD_VOLUME",
#                              "MAINTENANCE_SEVERITY", "MODIFIED_CONDITION_COUNT", "MULTIPLE_CONDITION_COUNT",
#                              "NODE_COUNT", "NORMALIZED_CYLOMATIC_COMPLEXITY",
#                              "NUM_OPERANDS", "NUM_OPERATORS", "NUM_UNIQUE_OPERANDS", "NUM_UNIQUE_OPERATORS",
#                              "NUMBER_OF_LINES", "PERCENT_COMMENTS", "LOC_TOTAL", "Defective"]
# 一
# df_example_noCols = pd.read_csv(filePath,header=None)
# df_example_noCols.columns = ['A', 'B','C']
# # 二、读数的同时添加标题
# df_example_noCols = pd.read_csv("CM1result.txt", header=None, names=['A', 'B','C'])







origdata = pd.read_csv("../../resource/csv/CM1.csv")
print(len(origdata.columns))
# # 选取数据（获取数据的列）
k = []
for i in range(0, len(origdata.columns)):
    k.append(origdata.columns.values[i])  # print(origdata.columns.values[1])#列名数据提取样式
# # 读数后加标题
df_example_noCols = pd.read_csv("PC1result.txt", header=None)
df_example_noCols.columns=k
df1=df_example_noCols
df1.to_csv("PC1result.csv",index=None)
print(df_example_noCols.columns)
