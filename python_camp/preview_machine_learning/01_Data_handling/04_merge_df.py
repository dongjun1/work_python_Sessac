import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

fpath = './dataset'

## create DataFrame..
exam_data = {
    '수학':[90, 80, 70],
    '영어':[90., 70., 90.],
    '성별':['남', '남', '여'],
    '합격':[True, False, True]
}

exam_df = pd.DataFrame(exam_data)
# print(exam_df)

# # Attribute
# print(exam_df.shape)
# print(exam_df.columns)
# print(exam_df.dtypes)
# print(exam_df.info())

# Summary statistics
# print(exam_df['수학'].mean())
# print(exam_df['수학'].median())
# print(exam_df['수학'].std())
# print(exam_df['수학'].var())
# print(exam_df['수학'].sem())
# print(exam_df['수학'].min())
# print(exam_df['수학'].max())
# print(exam_df['수학'].describe())

# seaborn
# print(sns.get_dataset_names()) -> check dataset's name
# tips = sns.load_dataset('tips')
# # print(tips.head())
# print('correlation = \n', tips[['total_bill', 'tip']].corr())
# plt.scatter(tips['total_bill'], tips['tip'], s = 100)
# plt.show()

# read csv
# fname = 'cars.csv'
# cars = pd.read_csv(os.path.join(fpath, fname))
# print(cars.head())

# # read excel
# excel = 'vor_r.xlsx'
# vor_r = pd.read_excel(os.path.join(fpath, excel))
# print(vor_r.head())

# exam DataFrame

df1 = pd.DataFrame({
    'a':['a0', 'a1', 'a2', 'a3'],
    'b':['b0', 'b1', 'b2', 'b3'],
    'c':['c0', 'c1', 'c2', 'c3']
}, index = [0, 1, 2, 3])

df2 = pd.DataFrame({
    'a':['a2', 'a3', 'a4', 'a5'],
    'b':['b2', 'b3', 'b4', 'b5'],
    'c':['c2', 'c3', 'c4', 'c5'],
    'd':['d2', 'd3', 'd4', 'd5'],
}, index = [2, 3, 4, 5])

# print('df1 = \n', df1)
# print('df2 = \n', df2)

# Concatenation
# concat_row = pd.concat([df1, df2], join = 'outer', ignore_index = True)
# print('concat row = \n', concat_row)

# concat_col = pd.concat([df1, df2], axis = 1, join = 'inner', ignore_index = False) # col로 병합 시 ignore index는 True,False에 영향을 받지 않음. 동작X
# print('cocat col = \n', concat_col)

# read exam stock data
# join도 index 기준으로 병합. 이름이 겹치는 col이 있다면 병합불가.
stock = pd.read_excel(os.path.join(fpath, 'stock price.xlsx'))
stock_valuation = pd.read_excel(os.path.join(fpath, 'stock valuation.xlsx'))

# stock.set_index('id', inplace = True)
# stock_valuation.set_index('id', inplace = True)
# # print('stock = \n', stock)
# # print('stock_valuation = \n', stock_valuation)

# stock_join = stock.join(stock_valuation, how = 'inner') # suffix : 접두사.
# print('stock_join_by_stock = \n', stock_join)

# merge DataFrame
merge_inner = pd.merge(stock, stock_valuation, how = 'inner', on = 'id')
# print('merge_inner = \n', merge_inner)

merge_outer = pd.merge(stock, stock_valuation, how = 'outer', on = 'id')
# print('merge_outer = \n', merge_outer)

merge_left = pd.merge(stock, stock_valuation, how = 'left', left_on = 'stock_name', right_on = 'name')
# print('merge_left = \n', merge_left)

merge_right = pd.merge(stock, stock_valuation, how = 'right', left_on = 'stock_name', right_on = 'name')
# print('merge_right = \n', merge_right)

# load dataset titanic of seaborn
titanic = sns.load_dataset('titanic')

front = titanic.loc[:, :'fare']
# print(front.head())
# print('fare mean = \n', front['fare'].mean().round(3))

# using filter
# pclass1 = front['pclass'] == 1
# pclass1_df = front[pclass1]
# pclass1_mean = pclass1_df['fare'].mean()

# pclass2 = front['pclass'] == 2
# pclass2_df = front[pclass2]
# pclass2_mean = pclass1_df['fare'].mean()

# pclass3 = front['pclass'] == 3
# pclass3_df = front[pclass3]
# pclass3_mean = pclass1_df['fare'].mean()

# using group by
# group_class = front.groupby('pclass')
# pclass_mean = group_class['fare'].mean()
# print('pclass mean = \n', pclass_mean)

# titanic dataset은 statistic의 기초 domain이 많이 필요한 dataset.
# group_survived = front.groupby(['survived', 'pclass'])
# survived_mean = group_survived[['fare', 'age', 'sibsp']].mean()
# print('survived mean = \n', survived_mean)

# group_sex = front.groupby('sex')
# sex_mean = group_sex['fare'].mean()
# print('sex mean = \n', sex_mean)

group_survived = front.groupby(['survived', 'pclass'])
# print(group_survived['sex'].value_counts())

# aggregation function
front.head()
print('mean = ', front['age'].mean())
print('median = ', front['age'].median())

print(front['age'].agg(['mean', 'std', 'median', 'min', 'max']))
print(group_survived['age'].agg(['mean', 'std', 'median', 'min', 'max']))
