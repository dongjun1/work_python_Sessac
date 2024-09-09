import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

fpath = './dataset'

# plot에 한글을 보이게 setting.
plt.rc('font', family = 'gulim')
plt.rc('axes', unicode_minus = False)

# plt.plot([1, 2, 3, 4])
# plt.title('한글')
# plt.show()

titanic = sns.load_dataset('titanic')
# print('titanic = \n', titanic.head())
# print(titanic.info())

# missing value
# print('missing value = ')
# print(titanic.isnull().sum())

titanic_df = titanic.loc[:, :'fare']
# print(titanic_df.info())
''' find missing value in "age"
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 7 columns):
 #   Column    Non-Null Count  Dtype
---  ------    --------------  -----
 0   survived  891 non-null    int64
 1   pclass    891 non-null    int64
 2   sex       891 non-null    object
 3   age       714 non-null    float64
 4   sibsp     891 non-null    int64
 5   parch     891 non-null    int64
 6   fare      891 non-null    float64
dtypes: float64(2), int64(4), object(1)
memory usage: 48.9+ KB
None
'''

titanic_df.dropna(axis = 0, inplace = True)
# print(titanic_df.info())
''' drop null in age
<class 'pandas.core.frame.DataFrame'>
Index: 714 entries, 0 to 890
Data columns (total 7 columns):
 #   Column    Non-Null Count  Dtype
---  ------    --------------  -----
 0   survived  714 non-null    int64
 1   pclass    714 non-null    int64
 2   sex       714 non-null    object
 3   age       714 non-null    float64
 4   sibsp     714 non-null    int64
 5   parch     714 non-null    int64
 6   fare      714 non-null    float64
dtypes: float64(2), int64(4), object(1)
memory usage: 44.6+ KB
None
'''

# sns.regplot : axis-level
# fig, ax = plt.subplots(figsize = (5, 5))
# sns.regplot(titanic_df, x = 'age', y = 'fare')
# plt.title('regression of titanic by age & fare')
# plt.show()
# fig.savefig('titanic_reg.png')

# order : 선형식의 차수를 의미함.
# sns.regplot(titanic_df, x = 'age', y = 'fare', 
#             fit_reg = True,
#             color = 'red', 
#             marker = '.',
#             order = 1, 
#             label = 'Titanic data',
#             scatter_kws = {'fc':'b', 'ec':'b', 's':100, 'alpha':0.3},
#             line_kws = {'lw':2, 'ls':'--', 'alpha':0.8})
# plt.xlabel('나이')
# plt.ylabel('요금')
# plt.title('나이 vs 요금')
# plt.legend(loc = 0)
# plt.show()

# load diamonds
diamonds = sns.load_dataset('diamonds')
# print(diamonds.columns)
'''
Index(['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'price', 'x', 'y',
       'z'],
      dtype='object')
'''
# print(diamonds.head())
# ideal = diamonds['cut'] == 'Ideal'
# price = diamonds[ideal].loc[:, 'price']
# carat = diamonds[ideal].loc[:, 'carat']
# sns.regplot(x = carat, y = price, order = 2, scatter_kws = {'fc':'b', 'ec':'b', 's':100, 'alpha':0.5}, line_kws = {'color':'red'})
# plt.title('carat vs price in diamonds cutting is Ideal')
# plt.show()

# sns.lmplot : figure-level
penguins = sns.load_dataset('penguins')
# print(penguins.head())
# # print(penguins['species'].value_counts())

# sns.lmplot(penguins, x = 'bill_length_mm', y = 'bill_depth_mm', hue = 'species', col = 'sex')
# plt.show()
# sns.lmplot(penguins, x = 'bill_length_mm', y = 'bill_depth_mm', col = 'species', row = 'sex', hue = 'island', sharex = False, sharey = False)
# plt.show()

# example..
# diamonds_plot = sns.lmplot(diamonds.sample(n = 5000), x = 'carat', y = 'price', hue = 'cut', col = 'clarity', col_wrap = 4)
# diamonds_plot.set_titles(size = 15)
# diamonds_plot.set_xlabels('Carat', size = 15)
# diamonds_plot.set_ylabels('Price', size = 15)
# plt.show()

# sns.scatterplot
tips = sns.load_dataset('tips')
# print('tips = \n', tips.head())
'''
    total_bill   tip     sex smoker  day    time  size
0       16.99  1.01  Female     No  Sun  Dinner     2
1       10.34  1.66    Male     No  Sun  Dinner     3
2       21.01  3.50    Male     No  Sun  Dinner     3
3       23.68  3.31    Male     No  Sun  Dinner     2
4       24.59  3.61  Female     No  Sun  Dinner     4
'''
# print(tips.info())
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 244 entries, 0 to 243
Data columns (total 7 columns):
 #   Column      Non-Null Count  Dtype
---  ------      --------------  -----
 0   total_bill  244 non-null    float64
 1   tip         244 non-null    float64
 2   sex         244 non-null    category
 3   smoker      244 non-null    category
 4   day         244 non-null    category
 5   time        244 non-null    category
 6   size        244 non-null    int64
dtypes: category(4), float64(2), int64(1)
memory usage: 7.4 KB
None
'''
# print(tips.columns) # Index(['total_bill', 'tip', 'sex', 'smoker', 'day', 'time', 'size'], dtype='object')
# sns.scatterplot(tips, x = 'total_bill', y = 'tip', hue = 'time', s = 100, alpha = 0.7, palette = 'Set3')
# plt.show()

# example..
# pd.options.mode.copy_on_write = True # remove_unused_catrgories()의 warning message를 출력하지 않기 위해 사용. 정확한 동작은 추후 확인 해봐야함.
# color_dia = (diamonds['color'] == 'D') | (diamonds['color'] == 'E') | (diamonds['color'] == 'F')
# cut_dia = (diamonds['cut'] == 'Ideal') | (diamonds['cut'] == 'Good') | (diamonds['cut'] == 'Fair')
# sample_dia = diamonds[color_dia & cut_dia]
# sample_dia['color'] = sample_dia['color'].cat.remove_unused_categories()
# sample_dia['cut'] = sample_dia['cut'].cat.remove_unused_categories()
# sns.scatterplot(sample_dia, x ='carat', y = 'price', hue = 'color', style = 'cut')
# plt.show()

# barplot
# fig, axs = plt.subplots(1, 2, figsize = (8, 4))
# sns.barplot(penguins, x = 'species', y = 'body_mass_g', hue = 'sex', estimator = 'mean', alpha = 0.6, errorbar = 'sd', ax = axs[0])
# axs[0].set_title('Mean')
# sns.barplot(penguins, x = 'species', y = 'body_mass_g', hue = 'sex', estimator = 'median', alpha = 0.6, errorbar = 'sd', ax = axs[1])
# axs[1].set_title('Median')
# plt.subplots_adjust(wspace = 0.4)
# plt.show()

# example..
iris = sns.load_dataset('iris')
# print(iris.head())
'''
   sepal_length  sepal_width  petal_length  petal_width species
0           5.1          3.5           1.4          0.2  setosa
1           4.9          3.0           1.4          0.2  setosa
2           4.7          3.2           1.3          0.2  setosa
3           4.6          3.1           1.5          0.2  setosa
4           5.0          3.6           1.4          0.2  setosa
'''
# y_list = iris.columns[:4]
# fig, axs = plt.subplots(1, 4, figsize = (16, 8))
# for i in range(4):
#     sns.barplot(iris, x = 'species', y = y_list[i], hue = 'species', ax = axs[i])

# # sns.barplot(iris, x = 'species', y = 'sepal_length', hue = 'species', ax = axs[0])
# # sns.barplot(iris, x = 'species', y = 'sepal_width', hue = 'species', ax = axs[1])
# # sns.barplot(iris, x = 'species', y = 'petal_length', hue = 'species', ax = axs[2])
# # sns.barplot(iris, x = 'species', y = 'petal_width', hue = 'species', ax = axs[3])
# plt.subplots_adjust(wspace = 0.6)
# plt.show()

# example...
mtcars = pd.read_excel(os.path.join(fpath, 'mtcars.xlsx'))
# print(mtcars.head())
'''
          Unnamed: 0   mpg  cyl   disp   hp  drat     wt   qsec  vs  am  gear  carb
0          Mazda RX4  21.0    6  160.0  110  3.90  2.620  16.46   0   1     4     4
1      Mazda RX4 Wag  21.0    6  160.0  110  3.90  2.875  17.02   0   1     4     4
2         Datsun 710  22.8    4  108.0   93  3.85  2.320  18.61   1   1     4     1
3     Hornet 4 Drive  21.4    6  258.0  110  3.08  3.215  19.44   1   0     3     1
4  Hornet Sportabout  18.7    8  360.0  175  3.15  3.440  17.02   0   0     3     2
'''
# print(mtcars.columns)
'''
Index(['Unnamed: 0', 'mpg', 'cyl', 'disp', 'hp', 'drat', 'wt', 'qsec', 'vs',
       'am', 'gear', 'carb'],
      dtype='object')
'''
# print(mtcars['cyl'].value_counts())
'''
cyl
8    14
4    11
6     7
Name: count, dtype: int64
'''
grouped_cyl = mtcars.groupby('cyl')
print(grouped_cyl['mpg'].agg(['mean', 'std']))
print('=' * 50)

grouped_am = mtcars.groupby('am')
print(grouped_am['mpg'].agg(['mean', 'std']))
print('=' * 50)

grouped_vs = mtcars.groupby('vs')
print(grouped_vs['mpg'].agg(['mean', 'std']))
print('=' * 50)

desc = mtcars.loc[:, ['mpg', 'wt']].describe()
print(desc[-1:-6:-1])
print('=' * 50)

print(mtcars.loc[:, ['cyl', 'am']].value_counts().unstack()) # unstack 한 번 찾아보기.
print('=' * 50)

kpg = mtcars['mpg'] * 1.61
mtcars.insert(2, 'kpg', kpg)
print(mtcars.head(7))
print('=' * 50)

# kpg_cyl = sns.barplot(mtcars, x = 'cyl', y = 'kpg', errorbar = None)
# kpg_cyl.bar_label(kpg_cyl.containers[0], fontsize=10)
# plt.xlabel('Cylinder number')
# plt.ylabel('Frequency (k/g)')
# plt.show()

# sns.boxplot(mtcars, x = 'am', y = 'mpg')
# plt.xlabel('Transmission')
# plt.ylabel('mpg')
# plt.show()

# sns.regplot(mtcars, x = 'wt', y = 'mpg')
# plt.xlabel('Weight')
# plt.ylabel('mpg')
# plt.show()

''' # 생각해보기. value count barplot by am and cyl
kpg_cyl_am = sns.barplot(mtcars, x = 'cyl', y = grouped_cyl['mpg'], hue = 'am', errorbar = None)
kpg_cyl_am.bar_label(kpg_cyl_am.containers[0], fontsize=10)
plt.show()
'''
