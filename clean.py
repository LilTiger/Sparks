# 由于使用的SCOP中包含TAPE远程同源检测fold_level的测试集 故在最开始clean掉这些数据 再划分训练集和测试集
import pandas as pd
import csv
import numpy as np
import tqdm
import json

csv_file = open('protein.csv')
reader = csv.reader(csv_file)
# read_csv方法默认会在开头加入新的unnamed列 设置index_col=0可以避免此现象
df = pd.read_csv('protein.csv', index_col=0)
# 重复的sequence数量和列表
n = 0
duplicate = []

with open("remote_homology_test_fold_holdout.json", 'r') as fp:
    json_data = json.load(fp)

    for index_1, row in tqdm.tqdm(df.iterrows()):
        # sequence = df['Sequence'].to_list()
        a = df.loc[index_1, 'Sequence']
        for index_2 in range(json_data.__len__()):
            b = json_data[index_2]['primary']
            if a == b:
                df.drop(index=index_1, inplace=True)
                duplicate.append(a)
                n += 1

print(n)
df.to_csv('protein_cleaned.csv')
