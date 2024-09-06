import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

fpath = './dataset'

# csv file read
fname = 'cars.csv'
file = os.path.join(fpath, fname)
# print(file)

# cars_df = pd.read_csv(file, header = 0, sep = ',')
# print(cars_df.head(10))

# fish_df = pd.read_csv('https://bit.ly/fish_csv')
# print(fish_df.head())

# excel file read
excel_name = 'mtcars.xlsx'
mtcars_df = pd.read_excel(os.path.join(fpath, excel_name))
# print(type(mtcars_df))
# print(mtcars_df.head())
# print(mtcars_df.tail())
# print(mtcars_df.info())
'''
RangeIndex: 32 entries, 0 to 31
Data columns (total 12 columns):
 #   Column      Non-Null Count  Dtype
---  ------      --------------  -----
 0   Unnamed: 0  32 non-null     object -> str or category
 1   mpg         32 non-null     float64
 2   cyl         32 non-null     int64
 3   disp        32 non-null     float64
 4   hp          32 non-null     int64
 5   drat        32 non-null     float64
 6   wt          32 non-null     float64
 7   qsec        32 non-null     float64
 8   vs          32 non-null     int64
 9   am          32 non-null     int64
 10  gear        32 non-null     int64
 11  carb        32 non-null     int64
dtypes: float64(5), int64(6), object(1)
memory usage: 3.1+ KB
None
'''

# attribute of mtcars
# print('index : \n', mtcars_df.index)
# print('shape : \n', mtcars_df.shape)
# print('dtypes : \n', mtcars_df.dtypes)
# print('columns : \n', mtcars_df.columns)
# print(mtcars_df.describe())
'''
             mpg        cyl        disp          hp       drat  ...       qsec         vs         am       gear     carb
count  32.000000  32.000000   32.000000   32.000000  32.000000  ...  32.000000  32.000000  32.000000  32.000000  32.0000
mean   20.090625   6.187500  230.721875  146.687500   3.596563  ...  17.848750   0.437500   0.406250   3.687500   2.8125
std     6.026948   1.785922  123.938694   68.562868   0.534679  ...   1.786943   0.504016   0.498991   0.737804   1.6152
min    10.400000   4.000000   71.100000   52.000000   2.760000  ...  14.500000   0.000000   0.000000   3.000000   1.0000
25%    15.425000   4.000000  120.825000   96.500000   3.080000  ...  16.892500   0.000000   0.000000   3.000000   2.0000
50%    19.200000   6.000000  196.300000  123.000000   3.695000  ...  17.710000   0.000000   0.000000   4.000000   2.0000
75%    22.800000   8.000000  326.000000  180.000000   3.920000  ...  18.900000   1.000000   1.000000   4.000000   4.0000
max    33.900000   8.000000  472.000000  335.000000   4.930000  ...  22.900000   1.000000   1.000000   5.000000   8.0000
'''
# cyl = mtcars_df['cyl'].astype('category')
# vs = mtcars_df['vs'].astype('category')
# am = mtcars_df['am'].astype('category')
# mtcars_df['cyl'] = cyl
# mtcars_df['vs'] = vs
# mtcars_df['am'] = am
# print(mtcars_df.describe())
# print(mtcars_df.info())

# create DataFrame by dict...
# data = {
#     'name' : ['홍길동', '임꺽정', '이순신'],
#     'algorithm' : ['A', 'A+', 'B'],
#     'basic' : ['C', 'B', 'B+'],
#     'python' : ['B+', 'C', 'C+']
# }
# # print(data)
# my_df = pd.DataFrame(data)
# print(my_df)

# create DataFrame by list.....
# shcool_list = [
#     [15, '남', '덕영중'],
#     [17, '여', '수리중']
# ]
# school_df = pd.DataFrame(shcool_list, columns = ['나이', '성별', '학교'])
# print(school_df)
# print(school_df.index)
# print(school_df.columns)

# summary statistics
# print(sns.get_dataset_names())
# mpg_df = sns.load_dataset('mpg')
# print(mpg_df.head())
# print('shape : ', mpg_df.shape)
# print('info : ', mpg_df.info())
# print('columns : ', mpg_df.columns)

# example...
# example_data = {
#     '수학' : [90, 80, 70],
#     '영어' : [98, 89, 95],
#     '성별' : ['남', '남', '여'],
#     '합격' : [True, False, True]
# }
# exam_df = pd.DataFrame(example_data)
# print(exam_df)
# print('shape = ', exam_df.shape)
# print('column names = ', exam_df.columns)
# print('ndim = ', exam_df.ndim)
# print('dtype = \n', exam_df.dtypes)

# Countinus
# mpg_df = sns.load_dataset('mpg')
# # print(mpg_df['mpg'].mean().round(3))
# # print(mpg_df['mpg'].median())
# # print(mpg_df['mpg'].var().round(3))
# # print(mpg_df['mpg'].std().round(3))
# # print(mpg_df['mpg'].sem().round(3)) # standard error of mean
# # print(mpg_df['mpg'].max())
# # print(mpg_df['mpg'].min())

# # # 이산형
# # print(mpg_df['cylinders'].value_counts())
# # print(mpg_df['origin'].value_counts())

# # ####
# # print(mpg_df[['mpg', 'weight']].mean(axis = 0).round(3))
# # print(mpg_df[['mpg', 'weight']].var(axis = 0).round(3))
# # print(mpg_df[['mpg', 'weight']].quantile(0.25))

# # correlation, x = 설명변수 (독립변수), y = 결과변수 (종속변수)
# print('correlation coefficient')
# print(mpg_df[['mpg', 'weight']].corr())

# # plt.scatter(mpg_df['mpg'], mpg_df['weight'])
# plt.scatter(mpg_df['weight'], mpg_df['mpg'])
# plt.title('weight vs mpg')
# plt.xlabel('weight (kg)')
# plt.ylabel('mpg (mile / gallon)')
# plt.show()
titanic = sns.load_dataset('titanic')
print(titanic.head())
print('*' * 100)
survived = titanic['survived']
# age = titanic['age']
# fare = titanic['fare']
print('survived = \n', survived.value_counts())
print('mean = \n', titanic[['age', 'fare']].mean())
print('std = \n', titanic[['age', 'fare']].std())

