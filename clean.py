# 由于使用的SCOP中包含TAPE远程同源检测fold_level的测试集 故在最开始clean掉这些数据 再划分训练集和测试集
import pandas as pd
import tqdm
import json

# read_csv方法默认会在开头加入新的unnamed列 设置index_col=0可以避免此现象
df = pd.read_csv('data/protein.csv', index_col=0)
# 重复的sequence数量和列表
n = 0
duplicate = []

with open("data/remote_homology_test_fold_holdout.json", 'r') as fp:
    json_data = json.load(fp)

    for index_1 in tqdm.tqdm(range(0, len(df))):
        # sequence = df['Sequence'].to_list()
        a = df.loc[index_1, 'Sequence']
        for index_2 in range(json_data.__len__()):
            b = json_data[index_2]['primary']
            if a == b:
                df.drop(index=index_1, inplace=True)
                duplicate.append(a)
                n += 1

print(n)
df.to_csv('data/protein_cleaned.csv', index=None)
