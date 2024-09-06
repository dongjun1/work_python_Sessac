import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dict_data = {
    'c0': np.arange(1, 4),
    'c1': np.arange(4, 7),
    'c2': np.arange(7, 10),
    'c3': np.arange(10, 13),
    'c4': np.arange(13, 16),
}

dict_df = pd.DataFrame(dict_data, index = ['r0', 'r1', 'r2'])
# print(dict_df)

# # index sort
# dict_df.sort_index(ascending = False, inplace = True)
# print('sort index of df :\n', dict_df)

# # value sort
# print('*' * 50)
# dict_df.sort_values(by = 'c0', ascending = True, inplace = True)
# print('sort value by c0 of df : \n', dict_df)

# broadcasting
# titanic = sns.load_dataset('titanic')
# print(titanic.head())
# fare_to_won = titanic['fare'] * 1300
# titanic.insert(7, 'won', fare_to_won)
# print(titanic.head())

# new_titanic = titanic.loc[:, 'fare':'won']
# print(new_titanic.head())
# print('*' * 50)
# add_num = new_titanic + 1000
# print(add_num.head())
# print(add_num - new_titanic)
fpath = './dataset'
fname = 'vor_r.xlsx'
vor_r = pd.read_excel(os.path.join(fpath,fname))
# print(vor_r.head())
'''
   group  id time    hz0.04    hz0.08     hz0.1    hz0.16    hz0.32    hz0.64       hz1
0      1   1  Pre  0.213840  0.717955  0.848727  0.850790  0.802738  0.808393  0.738967
1      1   2  Pre  0.345725  0.795440  0.781473  0.924972  0.801245  0.803880  0.781420
2      1   3  Pre  0.376285  0.799327  0.781350  0.777773  0.868007  0.868853  0.754252
3      1   4  Pre  0.451440  0.771083  0.808645  0.787825  0.839985  0.832352  0.818722
4      1   5  Pre  0.287360  0.825547  0.831303  0.818332  0.817320  0.816153  0.836298
'''
# mean_hz = vor_r.loc[:, 'hz0.04':'hz1'].mean(axis=1)
# mean_hz.sort_values(ascending=False, inplace=True)
# print(mean_hz.head())
'''
13    0.807488
7     0.805965
6     0.798594
8     0.794495
12    0.780265
dtype: float64
'''
# vor_r.insert(3, 'mean', mean_hz)
# vor_r.sort_values(ascending = False, by = 'mean', inplace = True)
# mean10_vor = vor_r.head(10)
# # print(mean10_vor)

# # filter
# mean10_vor.iloc[5:, 2] = 'Post'
# # print(mean10_vor)
# '''
#     group  id  time      mean    hz0.04    hz0.08     hz0.1    hz0.16    hz0.32    hz0.64       hz1
# 13      2  14   Pre  0.807488  0.387743  0.831845  0.856380  0.910937  0.900707  0.898062  0.866737
# 7       1   8   Pre  0.805965  0.385955  0.822715  0.845595  0.913647  0.907267  0.890155  0.876423
# 6       1   7   Pre  0.798594  0.397730  0.811073  0.800822  0.864213  0.914220  0.902707  0.899395
# 8       1   9   Pre  0.794495  0.369468  0.801775  0.831343  0.915945  0.902552  0.907822  0.832563
# 12      2  13   Pre  0.780265  0.331255  0.807625  0.838877  0.878798  0.896892  0.891240  0.817165
# 11      2  12  Post  0.777636  0.313582  0.701873  0.831662  0.842542  0.930743  0.944820  0.878230
# 5       1   6  Post  0.775674  0.366550  0.841505  0.841613  0.829140  0.880497  0.846647  0.823767
# 3       1   4  Post  0.758579  0.451440  0.771083  0.808645  0.787825  0.839985  0.832352  0.818722
# 14      2  15  Post  0.758175  0.317962  0.809910  0.763848  0.805748  0.853225  0.914925  0.841608
# 15      2  16  Post  0.753647  0.273288  0.711820  0.736360  0.915415  0.891255  0.884295  0.863095
# '''

# idx1 = mean10_vor.loc[:, 'time'] == 'Pre'
# idx2 = mean10_vor.loc[:, 'mean'] >= 0.77
# # print('time is Pre:\n',mean10_vor[idx1])
# # print('mean >= 0.79:\n',mean10_vor[idx2])
# # print('time is Pre and mean >= 0.77:\n', mean10_vor[(idx1) & (idx2)])
# # print('time is Pre or mean >= 0.77:\n', mean10_vor[(idx1) | (idx2)])
# print('time is Pre and mean >= 0.77 and group = 1:\n', mean10_vor[(idx1) & (idx2) & (mean10_vor.loc[:, 'group'] == 1)])

# dataset_names = sns.get_dataset_names() -> see the dataset names in seaborn
tips = sns.load_dataset('tips')
# print(tips.head())
# corr_bill_to_tip = tips.loc[:, 'total_bill':'tip'].corr()
# print('*' * 50)
# print('correlation coefficient = \n', corr_bill_to_tip)
# print('*' * 50)
# plt.scatter(tips['total_bill'], tips['tip'])
# plt.title('total_bill vs tip')
# plt.xlabel('total_bill')
# plt.ylabel('tip')
# print('Scatter Plot')
# plt.show()

mpg = sns.load_dataset('mpg')
# print(mpg.head())
kml = mpg['mpg'] * 0.425143707
mpg.insert(1, 'kml', kml)

print('mtcars mpg -> kmh = \n', mpg.head())