import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rc('font', family = 'gulim')
plt.rc('axes', unicode_minus = False)

stock = pd.read_excel('./dataset/stock price.xlsx')
stock_val = pd.read_excel('./dataset/stock valuation.xlsx')

# print('stock = \n', stock.head())
# print('stock_val = \n', stock_val.head())

# concat
# print(pd.concat([stock, stock_val], join = 'outer', ignore_index = True).head())
# print(pd.concat([stock, stock_val], axis = 1, join = 'outer', ignore_index = True).head())

# join
# print(stock.join(stock_val, lsuffix = 'x', rsuffix = 'y', how = 'inner').head())
# print(stock_val.join(stock, lsuffix = 'x', rsuffix = 'y', how = 'inner').head())

# merge
# print(pd.merge(stock, stock_val, left_on = 'stock_name', right_on = 'name', how = 'left').head())

# load titanic
titanic = sns.load_dataset('titanic')
# print(titanic.head())

titanic_group = titanic.groupby('survived')
# print(titanic_group[['age', 'fare']].mean())
# print(titanic_group[['age', 'fare']].std())

# print(titanic['age'].agg(['mean', 'median', 'std', 'min', 'max']))
# print(titanic_group['age'].agg(['mean', 'median', 'std', 'min', 'max']))

# regplot
# sns.regplot(titanic, x = 'age', y = 'fare', fit_reg = True, color = 'r', order = 1, label = 'Titanic')
# plt.title('나이 vs 요금')
# plt.xlabel('Age')
# plt.ylabel('Fare')
# plt.legend()
# plt.show()

# load penguin's data
penguins = sns.load_dataset('penguins')

# lmplot
# sns.lmplot(penguins, x = 'bill_length_mm', y = 'bill_depth_mm', hue = 'species', col = 'sex')
# sharex, sharey argument will change facet_kws.
# sns.lmplot(penguins, x = 'bill_length_mm', y = 'bill_depth_mm', col = 'species', row = 'sex', sharex = False, sharey = False)
# plt.show()

# scatterplot
sns.scatterplot(penguins, x = 'bill_length_mm', y = 'bill_depth_mm', s = 100, hue = 'species', style = 'island', palette = 'pastel')
plt.show()