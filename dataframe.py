import pandas as pd
import numpy as np

# 创建空表
empdata = pd.DataFrame()
print('empty dataframe \'empdata\':', empdata)

# 初始化数据
empdata['test']=[10,20,30,40]
print('empdata after adding \'test\' column:', empdata)


# 增加一行数据
# empdata.append({'column':value, 'column':value}, ignore_index=True)
# column为列名（加引号） value为数值 ignore_index=True语句必须存在 确保可以直接以字典的格式赋值
empdata = empdata.append({'test': 50}, ignore_index=True)
print(empdata)

# DataFrame.insert(loc, column, value, allow_duplicates=False)
# loc： 插入的列的索引（从0开始） column：列名（自己定） value：插入的值
# allow_duplicates=False 确保dataframe中只有一列 名为column的列;如果设置为true 则可以存在同名列
empdata.insert(1, 'exam', np.nan, allow_duplicates=False)
print('empdata after adding the same column \'test\' :\n', empdata)


# 判断数据是否在某一列当中
# if x in empdata['column'].values:
# 注意column为列索引的名称 需要加引号
# 后面加上.values 表示在column这一列的所有值中寻找; 如果不加.values 会出现语法错误
if 10 in empdata['test'].values:
    print('10 is in \'text\' column')

