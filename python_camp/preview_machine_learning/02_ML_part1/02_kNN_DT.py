import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

plt.rc('font', family = 'gulim')
plt.rc('axes', unicode_minus = False)

# mtcars = pd.read_excel('../01_Data_handling/dataset/mtcars.xlsx')
# print(mtcars.head())

# train_X = mtcars.loc[:, 'cyl':'qsec']
# train_y = mtcars.loc[:, 'mpg']

# # print(train_X.head())
# # print(train_y.head())

# # LinearRegression
# lr = LinearRegression()
# lr.fit(train_X, train_y)
# # print('train Rsquare = ', round(lr.score(train_X, train_y), 3))

# coef_df = pd.DataFrame(lr.coef_, index = train_X.columns, columns = ['coefficient'])
# print(coef_df)

# kNN (k-nearest neighbors classfier)
# 도미자료
bream_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0]
bream_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0]

# 방어자료
smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

# print(f'도미 무게 = {round(np.mean(bream_weight), 4)}, 길이 = {round(np.mean(bream_length), 4)}')
# print(f'방어 무게 = {round(np.mean(smelt_weight), 4)}, 길이 = {round(np.mean(smelt_length), 4)}')

# scatter
# plt.scatter : input data가 1차원, sns.scatter : input data가 DF
# plt.scatter(bream_length, bream_weight, label = '도미')
# plt.scatter(smelt_length, smelt_weight, label = '방어')
# plt.title('도미 vs 방어')
# plt.xlabel('길이(cm)')
# plt.ylabel('무게(g)')
# plt.grid(linestyle = ':')
# plt.legend()
# plt.show()

 
length = bream_length + smelt_length # (49,)
weight = bream_weight + smelt_weight # (49,)

# x_2d
fish_data = np.column_stack([length, weight]) # np.column_stack 찾아보기.
# print(fish_data)
# print(type(fish_data)) # np.ndarray
# print(np.shape(fish_data)) # (49, 2)

# 도미를 1로 target
fish_target = np.append(np.ones(len(bream_length)), np.zeros(len(smelt_length)))

# kNN
# knn = KNeighborsClassifier(n_neighbors = 5, weights = 'uniform', metric = 'minkowski', p = 2) # default parameters
# knn.fit(fish_data, fish_target)
# print('train Accuracy = ', knn.score(fish_data, fish_target))

# # Attribute
# print('classes = ', knn.classes_)
# print('metric = ', knn.effective_metric_)
# print('samples = ', knn.n_samples_fit_)

# train / test split manual
# idx = np.arange(49)
# np.random.seed(911)
# np.random.shuffle(idx)

# train_X = fish_data[idx[:35]]
# train_y = fish_target[idx[:35]]

# test_X = fish_data[idx[35:]]
# test_y = fish_target[idx[35:]]

# print(np.column_stack([test_X, test_y]))

# plot
# plt.scatter(train_X[:, 0], train_X[:, 1], label = 'Train')
# plt.scatter(test_X[:, 0], test_X[:, 1], label = 'Test')
# plt.legend()
# plt.grid(linestyle = ':')
# plt.show()

# kNN
# knn = KNeighborsClassifier(n_neighbors = 5, weights = 'uniform', metric = 'minkowski', p = 2)
# knn.fit(train_X, train_y)
# print('Test Accuracy = ', knn.score(test_X, test_y))
# print('classes = ', knn.classes_)
# print('samples = ', knn.n_samples_fit_)
# print('metric = ', knn.effective_metric_)
# print('metric_params = ', knn.effective_metric_params_)

# train / test split with package
# train_X, test_X, train_y, test_y = train_test_split(fish_data, fish_target, stratify = fish_target)
# # print(train_X)
# knn = KNeighborsClassifier()
# knn.fit(train_X, train_y)
# print('Test Accuracy = ', knn.score(test_X, test_y))

# plot
# plt.scatter(train_X[:, 0], train_X[:, 1], label = 'Train')
# plt.scatter(25, 150, marker = '^', s = 200, label = 'New')
# plt.legend()
# plt.grid(linestyle = ':')
# plt.show()

# print(knn.predict([[25, 150]])) # 0으로 방어로 예측. why? 길이와 무게의 단위가 달라서. 따라서 kNN은 데이터의 표준화가 중요함.

# distance, index = knn.kneighbors([[25, 150]])
# plt.scatter(train_X[:, 0], train_X[:, 1], label = 'Train')
# plt.scatter(25, 150, marker = '^', s = 200, label = 'New')
# plt.scatter(train_X[index, 0], train_X[index, 1], c = 'r')
# plt.legend()
# plt.grid(linestyle = ':')
# plt.show()

# standardization
# x_mean = np.mean(train_X, axis = 0)
# x_std = np.std(train_X, axis = 0)

# train_scaled = (train_X - x_mean) / x_std
# test_scaled = (test_X - x_mean) / x_std
# # print(train_scaled)
# new = ([25, 150] - x_mean) / x_std
# plt.scatter(train_scaled[:, 0], train_scaled[:, 1])
# plt.scatter(new[0], new[1], marker = '^', s = 100)
# plt.grid(linestyle = ':')
# plt.show()

# knn.fit(train_scaled, train_y)
# print('Test Accuracy = ', knn.score(test_scaled, test_y))
# print('Prediction = ', knn.predict([new]))

# distance, index = knn.kneighbors([new])
# plt.scatter(train_X[:, 0], train_X[:, 1], label = 'Train')
# plt.scatter(new[0], new[1], label = 'new')
# plt.scatter(train_X[index, 0], train_X[index, 1], c = 'r')
# plt.legend()
# plt.grid(linestyle = ':')
# plt.show()

# GridSearch
# knn = KNeighborsClassifier()

# score_list = []
# n_list = []

# for n in range(3, 30):
#     if n % 2 != 0:
#         knn.n_neighbors = n
#         knn.fit(train_scaled, train_y)
#         n_list.append(n)
#         score_list.append(knn.score(test_scaled, test_y))

# plt.plot(n_list, score_list)
# plt.xlabel('k neighbors')
# plt.ylabel('Test Accuracy')
# plt.title('Grid Search of Fish data')
# plt.grid(linestyle = ':')
# plt.show()

# kNN을 이용하여 iris 자료 분류
# iris = sns.load_dataset('iris')
# # print(iris.head())
# '''
#    sepal_length  sepal_width  petal_length  petal_width species
# 0           5.1          3.5           1.4          0.2  setosa
# 1           4.9          3.0           1.4          0.2  setosa
# 2           4.7          3.2           1.3          0.2  setosa
# 3           4.6          3.1           1.5          0.2  setosa
# 4           5.0          3.6           1.4          0.2  setosa
# '''
# # print(iris['species'].value_counts())
# '''
# species
# setosa        50
# versicolor    50
# virginica     50
# ''' 
# X = iris.loc[:, 'sepal_length':'petal_width']
# y = iris.loc[:, 'species']


# train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.3, stratify = y)

# # Standardizaion
# x_mean = np.mean(X, axis = 0)
# x_std = np.std(X, axis = 0)
# train_scaled = (train_X - x_mean) / x_std
# test_scaled = (test_X - x_mean) / x_std

# # Visualization
# sns.pairplot(iris, hue = 'species')
# plt.show()

# knn = KNeighborsClassifier(n_neighbors = 7)
# knn.fit(train_X, train_y)
# print('not standardization Accuracy = ', knn.score(test_X, test_y))
# print('Predict = ', knn.predict([[3.1, 2.5, 0.8, 0.6]]))


# knn1 = KNeighborsClassifier(n_neighbors = 7)
# knn1.fit(train_scaled, train_y)
# print('standardization Accuracy = ', knn1.score(test_scaled, test_y))
# print('Predict = ', knn1.predict([[3.1, 2.5, 0.8, 0.6]]))

# print('classes = ', knn.classes_)
# print('feature name = ', knn.feature_names_in_)
# print('metric = ', knn.effective_metric_)
# print('samples = ', knn.n_samples_fit_)
# print('k = ', knn.n_neighbors)

# score_list = []
# n_list = []

# for n in range(3, 30):
#     if n % 2 != 0:
#         knn.n_neighbors = n
#         knn.fit(train_scaled, train_y)
#         n_list.append(n)
#         score_list.append(knn.score(test_scaled, test_y))

# plt.plot(n_list, score_list, color = 'r')
# plt.xlabel('k neighbors')
# plt.ylabel('Test Accuracy')
# plt.title('Grid Search of Iris data')
# plt.grid(linestyle = ':')
# plt.show()

wine = pd.read_csv('https://bit.ly/wine-date')
# print(wine.head())
'''
   alcohol  sugar    pH  class
0      9.4    1.9  3.51    0.0
1      9.8    2.6  3.20    0.0
2      9.8    2.3  3.26    0.0
3      9.8    1.9  3.16    0.0
4      9.4    1.9  3.51    0.0
'''
# print(wine['class'].value_counts())
'''
class
1.0    4898
0.0    1599
Name: count, dtype: int64
'''
# class 는 float형 일 이유가 X
wine['class'] = wine['class'].astype('int32')
# print(wine['class'].value_counts())
'''
class
1    4898
0    1599
Name: count, dtype: int64
'''

wine_group = wine.groupby('class', observed = True)

for idx, data in wine_group:
    print('class = ', idx)
    print(data.describe())
    print('=' * 50)

# Visualizaion
sns.countplot(wine, x = 'class')
plt.show()


# choose X / y
X = wine.loc[:, 'alcohol':'pH']
y = wine['class']

# split train / test
train_X, test_X, train_y, test_y = train_test_split(X, y, stratify = y)

# standardization
x_mean = np.mean(train_X, axis = 0)
x_std = np.std(train_X, axis = 0)
train_scaled = (train_X - x_mean) / x_std
test_scaled = (test_X - x_mean) / x_std

# DT
# default of DT : entropy 값이 가장 커지는 기준으로 나눔.
# criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_impurity_decrease=0.0
# min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, max_leaf_nodes=None, class_weight=None
# dt의 GridSearch는 depth, num of terminal nodes, num of features, feature가 연속형이면 하나하나 전부.... 등을 고려해야함.

dt = DecisionTreeClassifier()
dt.fit(train_scaled, train_y)
print('Test Accuracy of DT = ', dt.score(test_scaled, test_y))

# kNN
knn = KNeighborsClassifier()
knn.fit(train_scaled, train_y)
print('Test Accuracy of kNN = ', knn.score(test_scaled, test_y))

n_list = []
score_list = []
for n in range(3, 30):
    if n % 2 != 0:
        knn.n_neighbors = n
        knn.fit(train_scaled, train_y)
        score_list.append(knn.score(test_scaled, test_y))
        n_list.append(n)

plt.plot(n_list, score_list, color = 'r')
plt.xlabel('k neighbors')
plt.ylabel('Test Accuracy')
plt.title('Grid Search of wine data')
plt.grid(linestyle = ':')
plt.show()