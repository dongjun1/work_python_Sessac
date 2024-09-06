import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

fpath = './dataset'

exam_data = {
    '수학':[90, 80, 90],
    '영어':[98, 89, 95],
    '성별':['남', '남', '여'],
    '합격':[True, False, True]
}

exam_df = pd.DataFrame(exam_data)
# # print(exam_df)
# # print(exam_df.shape)
# # print(exam_df.info())

# list_data = [
#     ['green', 'M', 13.5, 'class1'],
#     ['red', 'L', 15.3, 'class2'],
#     ['blue', 'XL', 10.1, 'class1']
# ]

# list_df = pd.DataFrame(list_data, columns = ['color', 'size', 'price', 'class_label
# '])
# # print(list_df)

# # 변수(column)명 바꾸기
# list_df.rename(columns = {'color':'색상', 'size':'사이즈', 'price':'가격', 'class_label':'클래스'}, index = {0:1, 1:2, 2:3}, inplace = True)
# # print(list_df)

# print('origin of exam_df : \n', exam_df)
# exam_df.drop(labels = '수학', axis = 1, inplace = True)
# print('drop col of "수학": \n', exam_df)
# exam_df.drop(labels = ['영어', '합격'], axis = 1, inplace = True)
# print('drop col of "영어", "합격" : \n', exam_df)
# exam_df.drop(labels = [0], axis = 0, inplace = True)
# print('drop row of 0 : \n', exam_df)

fname = 'vor_r.xlsx'
vor_r = pd.read_excel(os.path.join(fpath,fname))
# print('*' * 100)
# print('vor = \n', vor_r.head())
# vor_r.rename(columns = {'group':'소속', 'id':'번호'}, inplace = True)
# print('*' * 100)
# print('vor = \n', vor_r.head())
# vor_r.drop(labels = ['소속', '번호'], axis = 1, inplace = True)
# print('*' * 100)
# print('vor = \n', vor_r)

# handling missing values

# print('*' * 50)
# print('exam_df = \n', exam_df)
# exam_df.loc[0, '성별'] = np.nan
# exam_df.loc[1, '수학':'영어'] = None
# print('*' * 50)
# print('exam_df = \n', exam_df)
# exam_df.dropna(axis = 1, inplace = True)
# print('*' * 50)
# print('exam_df = \n', exam_df)
# exam_df.dropna(axis = 0, subset = ['수학'], inplace = True)
# print('*' * 50)
# print('exam_df = \n', exam_df)

# data indexing
# indexing of col name
# print('col name of 수학 : \n', exam_df['수학'])
# print('col name of 수학, 성별 : \n', exam_df[['수학', '성별']])

# indexing of slicing
# print(exam_df[2:3])

# indexing of loc -> dataframe의 idex를 사용.
# print(exam_df)
# print('*' * 50)
# print(exam_df.loc[0:2])
# print('*' * 50)
# print(exam_df.loc[:, ['수학', '합격']])
# print('*' * 50)
# print(exam_df.loc[1:2,'수학':'성별'])

# example..
df = pd.DataFrame({
    'hdz':['yes', 'no', 'no', 'no', 'yes'],
    'chestpain':[True, False, False, True, False],
    'cholesterol':[208, 282, 235, 277, 280],
    'sysbp':[160, 140, 188, 162, 122]
}, index = np.arange(1, 6))

# print('*' * 50)
# print(df)
# print('*' * 50)
# print(df.loc[[3, 5], 'chestpain'])
# print('*' * 50)
# print(df.loc[2:4, 'chestpain':'sysbp'])
# print('*' * 50)
# print(df.loc[:, 'hdz'])
# print('*' * 50)
# print(df.loc[:, ::-1])

# example by iloc -> python의 index를 사용
# print('*' * 50)
# print(df)
# print('*' * 50)
# print(df.iloc[:, 1:3])

# add few col
# exam_df['음악'] = [100, 90, 80]
# # print(exam_df)

# exam_df.insert(1, '코딩', [50, 80, 100])
# # print(exam_df)

# # add few row
# exam_df.loc[3] = [60, 90, 80, '여', False, 100]
# # print(exam_df)

# # change value
# exam_df.loc[3, '코딩'] = 0
# exam_df.loc[2, '코딩':'영어'] = 0
# print(exam_df)

# example change value...
# df.loc[5, 'sysbp'] = 10000
# print('change by loc:\n', df)
# df.iloc[2, 3] = 10000
# print('change by iloc:\n', df)
# df.loc[2:4, 'sysbp'] = [1, 2, 3] 
# print('change by loc:\n', df)

dict_data = {
    "c0" : np.arange(1, 4),
    "c1" : np.arange(4, 7),
    "c2" : np.arange(7, 10),
    "c3" : np.arange(10, 13),
    "c4" : np.arange(13, 16)
}
dict_df = pd.DataFrame(dict_data, index = ['r0', 'r1', 'r2'])
# print(dict_df)

# print('*' * 50)
# dict_df.set_index("c0", inplace = True)
# print(dict_df)
# print('*' * 50)
# dict_df.reset_index(inplace = True)
# print(dict_df)

print('vor = \n', vor_r.head())
print('*' * 50)
vor_r.set_index('id', inplace = True)
print('index id vor = \n', vor_r.head())
vor_r.reset_index(inplace = True)
