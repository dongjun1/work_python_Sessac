import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# print(sklearn.__version__) # 1.5.1

# Linear Regression without package
# np.random.seed(910)
# x = np.arange(1, 30, 2)
# y = 2 * x + 1 # coeff = 2, intercept = 1
# y_random = y + np.random.normal(0, 8, len(y))

# # plt.scatter(x, y, s = 50, label = 'y')
# # plt.scatter(x, y_random, s = 50, label = 'y_random')
# # plt.legend()
# # plt.show()

# # OLS : 최소제곱추정량
# # y = 2x + 1 , coeff, b1, 회귀계수 : 2, intercept, b0, 절편 : 1
# x_mean = np.mean(x)
# y_mean = np.mean(y_random)
# Sxy = sum((x - x_mean) * (y_random - y_mean))
# Sxx = sum((x - x_mean) ** 2)
# b1 = Sxy / Sxx
# b0 = y_mean - b1 * x_mean
# # print('b1 = ', b1)
# # print('b0 = ', b0)
# # print('y_random = ', y_random.round(1))
# # print('yhat = ', (b1 * y_random + b0).round(1))
# y_hat = b1 * x + b0

# plt.scatter(x, y, s = 50, label = 'y')
# plt.scatter(x, y_random, s = 50, label = 'y_random')
# plt.plot(x, y_hat, label = 'y_hat')
# plt.legend()
# plt.show()

# Linear Regression with package
# x_2d = x.reshape(-1, 1)
# print(x)
# print(x_2d)
# print(x_2d.shape)

# create class instance
# lr = LinearRegression()

# fitting
# lr.fit(x_2d, y_random)

# # print
# print('coeff = ', lr.coef_)
# print('intercept = ', lr.intercept_)

# # score : qof(quality of fit)
# # training set을 사용하여 Rsquare를 도출했기 때문에. training score라고 명시해줘야 함.
# print(lr.score(x_2d, y_random)) # Rsquare

# example..
# cars = pd.read_csv('../01_Data_handling/dataset/cars.csv')
# # print(cars.head())
# '''
#    speed  dist
# 0      4     2
# 1      4    10
# 2      7     4
# 3      7    22
# 4      8    16
# '''
# # print(cars.shape) # (50, 2)
# speed = cars['speed']
# dist = cars['dist']

# # Summary Statistic
# # print('cars info = \n', cars.info())
# # print('=' * 50)
# # print('describe of speed = \n', speed.describe())
# # print('=' * 50)
# # print('describe of dist = \n', dist.describe())
# # print('=' * 50)
# # print('corr = \n', cars[['speed', 'dist']].corr())
# # print('=' * 50)
# print('summary statistics : \n', cars.agg(['mean', 'std', 'median', 'min', 'max']).round(3))

# # histogram
# fig, axs = plt.subplots(1, 2, figsize = (10, 6))
# sns.histplot(cars, x = 'dist', ax = axs[0], kde = True)
# sns.histplot(cars, x = 'speed', ax = axs[1], kde = True)
# plt.show()

# # boxplot
# fig, axs = plt.subplots(1, 2, figsize = (10, 6))
# sns.boxplot(cars, y = 'speed', ax = axs[0])
# sns.stripplot(cars, y = 'speed', ax = axs[0])
# sns.boxplot(cars, y = 'dist', ax = axs[1])
# sns.stripplot(cars, y = 'dist', ax = axs[1])
# plt.show()

# # scatter
# sns.scatterplot(cars, x = 'speed', y = 'dist', s = 100)
# plt.title('Speed vs Stopping distance')
# plt.show()
# print plot
# sns.regplot(x = speed, y = dist)
# plt.title('Speed vs Dist')
# plt.xlabel('speed')
# plt.ylabel('dist')
# plt.show()

# Simple LinearRegression
# lr = LinearRegression()
# speed_arr = np.array(speed) # cars['speed'].values -> np.array, cars[['speed']] -> pd.DataFrame
# speed_arr = speed_arr.reshape(-1, 1)
# lr.fit(speed_arr, dist)
# print('coeff = ', lr.coef_.round(3)) # 3.932
# print('intercept = ', lr.intercept_.round(3)) # -17.579
# print('Rsquare = ', round(lr.score(speed_arr, dist), 3)) # 0.651
# # 결론 : 속도 1당 제동거리가 3.932만큼 증가함.

# # scatter
# x_line = np.linspace(5, 25, 10)
# plt.scatter(x = 'speed', y = 'dist', data = cars, s = 100)
# plt.plot(x_line, 3.932 * x_line - 17.579, c = 'r')
# plt.show()

# Multiple LinearRegression
mtcars = pd.read_excel('../01_Data_handling/dataset/mtcars.xlsx')
# print(mtcars.head())
'''
          Unnamed: 0   mpg  cyl   disp   hp  drat     wt   qsec  vs  am  gear  carb
0          Mazda RX4  21.0    6  160.0  110  3.90  2.620  16.46   0   1     4     4
1      Mazda RX4 Wag  21.0    6  160.0  110  3.90  2.875  17.02   0   1     4     4
2         Datsun 710  22.8    4  108.0   93  3.85  2.320  18.61   1   1     4     1
3     Hornet 4 Drive  21.4    6  258.0  110  3.08  3.215  19.44   1   0     3     1
4  Hornet Sportabout  18.7    8  360.0  175  3.15  3.440  17.02   0   0     3     2
'''
mpg = mtcars['mpg']
x = mtcars.loc[:, 'cyl':'wt']
print('summary statistics : \n', x.agg(['mean', 'std', 'median', 'min', 'max']).round(3))
print('mpg.shape = ', mpg.shape)
print('x.shape = ', x.shape)
x_list = ['cyl', 'disp', 'hp', 'drat', 'wt']

# corr
for elem in x_list:
    print('=' * 50)
    print(mtcars[['mpg', f'{elem}']].corr())


# scatter
fig, axs = plt.subplots(1, 5, figsize = (10, 6))
for i in range(5):
    sns.scatterplot(x = x.iloc[:, i], y = mpg, s = 100, ax = axs[i])
    plt.subplots_adjust(wspace = 0.6)
plt.show()

# boxplot
# fig, axs = plt.subplots(1, 5, figsize = (10, 6))
# for i in range(5):
#     sns.boxplot(x, y = x.iloc[:, i], ax = axs[i])
#     sns.stripplot(x, y = x.iloc[:, i], ax = axs[i])
#     plt.subplots_adjust(wspace = 0.6)
# plt.show()


lr = LinearRegression()
lr.fit(x, mpg)
print('coef = ', lr.coef_)
print('intercept = ', lr.intercept_)
print('Rsquare = ', lr.score(x, mpg))
