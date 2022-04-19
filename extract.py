# # 利用切片的思想从原始scop parse文件中分离数据
# import pandas as pd
# import numpy as np
#
# id_fa = []
# id_sf = []
# cf_list = []
# sf_list = []
# fa_list = []
#
# with open('scop-cla-latest.txt', 'r+') as fd:
#     for text in fd.readlines():
#         id_fa.append(text.split(' ')[0])
#         id_sf.append(text.split(' ')[5])
#         scop = text.strip().split(' ')[10].split(',')
#         # 继续在scop中提取CF SF FA
#         scop_cf = scop[2].split('=')[1]
#         cf_list.append(scop_cf)
#         scop_sf = scop[3].split('=')[1]
#         sf_list.append(scop_sf)
#         scop_fa = scop[4].split('=')[1]
#         fa_list.append(scop_fa)
#         print(scop)
# df = pd.DataFrame({'Family_id': id_fa, 'SuperFamily_id': id_sf, 'Sequence': '', 'CF': cf_list, 'SF': sf_list, 'FA': fa_list})
# df.to_csv('protein_pre.csv')
# print(df)

# 利用正则表达式寻找家族id对应sequence
import re
import pandas as pd
import csv
import numpy as np
import tqdm

csv_file = open('protein_pre.csv')
reader = csv.reader(csv_file)
df = pd.read_csv('protein_pre.csv')

# 将id和sequence合并到一行 生成新列表 便于后续匹配
new = []
with open('scop_sf_represeq_lib_latest.fa.txt') as fd:
    lines = fd.readlines()
    for i in range(lines.__len__()):
        if i % 2 == 0:
            new.append(lines[i].strip() + ' ' + lines[i+1])

for item in tqdm.tqdm(reader):
    if reader.line_num == 1:
        continue
    for index in range(new.__len__()):
        # 定位SuperFamily_id
        super_id = new[index].strip('>').split(' ')[0]
        if item[2] == super_id:
            sequence = new[index].strip().split(' ')[4]
            df['Sequence'] = df['Sequence'].astype(np.str)
            # 按照源protein.csv中SuperFamily_id顺序查找 故可按照 索引 依次赋予找到的 sequence
            df.at[int(item[0]), 'Sequence'] = sequence
            # print(df)
            # 找到之后及时跳出循环
            break
df.to_csv('protein.csv')


