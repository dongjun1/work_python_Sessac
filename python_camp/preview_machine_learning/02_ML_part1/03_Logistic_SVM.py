import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


plt.rc('font', family = 'gulim')
plt.rc('axes', unicode_minus = False)

# iris = sns.load_dataset('iris')
# # print(iris.head())

# # X = iris.loc[:, 'sepal_length':'petal_width']
# # y = iris['species']

# # train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.2, stratify = y)

# # # StandardScaler
# # ss = StandardScaler()
# # ss.fit(train_X)

# # train_scaled = ss.transform(train_X)
# # test_scaled = ss.transform(test_X)

# # ss.fit_transfrom()
# # fit과 transfrom을 한번에 수행하여 주지만 수행되어 구해진 parameter(mean, std 등)를 저장하고 있지 않기 때문에 동일한 parameter를 test set에 적용할 수 없음.

# # kNN
# # knn = KNeighborsClassifier()
# # knn.fit(train_scaled, train_y)
# # print(f'Train Accuracy = {round(knn.score(train_scaled, train_y), 4)}')
# # print(f'Test Accuracy = {round(knn.score(test_scaled, test_y), 4)}')

# # # Attribute
# # print(f'classes = {knn.classes_}')
# # print(f'metric = {knn.effective_metric_}')
# # # print(f'feature names = {knn.feature_names_in_}') : input이 DF일 경우에만 존재.
# # print(f'sample = {knn.n_samples_fit_}')
# # print(f'k = {knn.n_neighbors}')

# # DT
# wine = pd.read_csv('https://bit.ly/wine-date')
# # print(wine.head())
# '''
#    alcohol  sugar    pH  class
# 0      9.4    1.9  3.51    0.0
# 1      9.8    2.6  3.20    0.0
# 2      9.8    2.3  3.26    0.0
# 3      9.8    1.9  3.16    0.0
# 4      9.4    1.9  3.51    0.0
# '''
# # print(wine.info())
# '''
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 6497 entries, 0 to 6496
# Data columns (total 4 columns):
#  #   Column   Non-Null Count  Dtype
# ---  ------   --------------  -----
#  0   alcohol  6497 non-null   float64
#  1   sugar    6497 non-null   float64
#  2   pH       6497 non-null   float64
#  3   class    6497 non-null   float64
# dtypes: float64(4)
# memory usage: 203.2 KB
# None
# '''

# # change type to category
# wine['class'] = wine['class'].astype('int32').astype('category')
# # print(wine.info())
# '''
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 6497 entries, 0 to 6496
# Data columns (total 4 columns):
#  #   Column   Non-Null Count  Dtype
# ---  ------   --------------  -----
#  0   alcohol  6497 non-null   float64
#  1   sugar    6497 non-null   float64
#  2   pH       6497 non-null   float64
#  3   class    6497 non-null   category
# dtypes: category(1), float64(3)
# memory usage: 158.8 KB
# None
# '''
# # print(wine.head())
# '''
#    alcohol  sugar    pH class
# 0      9.4    1.9  3.51     0
# 1      9.8    2.6  3.20     0
# 2      9.8    2.3  3.26     0
# 3      9.8    1.9  3.16     0
# 4      9.4    1.9  3.51     0
# '''

# # Visualizaion
# # sns.boxplot(wine, x = 'class', y = 'sugar')
# # plt.ylim(0, 35)
# # plt.show()

# # sns.boxplot(wine, x = 'class', y = 'alcohol')
# # plt.ylim(5, 20)
# # plt.show()

# # sns.boxplot(wine, x = 'class', y = 'pH')
# # plt.ylim(0, 10)
# # plt.show()

# X = wine.iloc[:, :-1]
# y = wine.iloc[:, -1]

# # Split
# train_X, test_X, train_y, test_y = train_test_split(X, y, stratify = y)

# # Scaling
# ss = StandardScaler()
# ss.fit(train_X)

# # check params
# # print(ss.mean_)
# # print(ss.scale_)

# train_scaled = ss.transform(train_X)
# test_scaled = ss.transform(test_X)

# # DT
# dt = DecisionTreeClassifier(criterion = 'gini')
# dt.fit(train_scaled, train_y)

# # overfitting....
# # print(f'Train Accuracy = {round(dt.score(train_scaled, train_y), 3)}')
# # print(f'Test Accuracy = {round(dt.score(test_scaled, test_y), 3)}')
# '''
# Train Accuracy = 0.997
# Test Accuracy = 0.862
# '''

# # set max_depth, leaf node
# dt_modify = DecisionTreeClassifier(criterion = 'gini', max_depth = 5, max_leaf_nodes = 10)
# dt_modify.fit(train_scaled, train_y)
# # print(f'Train Accuracy = {round(dt_modify.score(train_scaled, train_y), 3)}')
# # print(f'Test Accuracy = {round(dt_modify.score(test_scaled, test_y), 3)}')
# '''
# Train Accuracy = 0.86
# Test Accuracy = 0.872
# '''

# # Visualizaion original DT
# # plot_tree(dt, max_depth = 3, filled = True)
# # plt.show()

# # # Visualizaion limit node and depth
# # plot_tree(dt_modify, filled = True)
# # plt.show()

# # print(dt_modify.feature_importances_)
# # df = pd.DataFrame(dt_modify.feature_importances_, index = train_X.columns)
# # print(df) 
# ''' depth 와 leaf 갯수를 제한 할 경우 sugar의 가중치가 훨씬 높아짐.
#                 0
# alcohol  0.142920
# sugar    0.745921
# pH       0.111158
# '''

# # print(dt.feature_importances_)
# # df = pd.DataFrame(dt.feature_importances_, index = train_X.columns)
# # print(df)
# '''
#                 0
# alcohol  0.234996
# sugar    0.516640
# pH       0.248364
# '''

# # 교차검증
# # cv_train_result = cross_validate(dt_modify, train_scaled, train_y, cv = 5)
# # print(f'split X = {cv_train_result["test_score"].mean()}')

# X_scaled = ss.fit_transform(X)
# # cv_X_result = cross_validate(dt_modify, X_scaled, y, cv = 5)
# # print(f'not split X = {cv_X_result["test_score"].mean()}')
# '''
# split X = 0.8594039909440321
# not split X = 0.8533217267720733
# '''

# # 현재 wine의 data가 75:25 정도로 데이터의 비율이 맞지 않기 때문에 이 비율을 반영하기 위해 StratifiedKFold를 사용. 
# splitter = StratifiedKFold(n_splits = 5, shuffle = True)
# # cv_result = cross_validate(dt_modify, X_scaled, y, cv = splitter)
# # print(f'not split X with splitter = {cv_result["test_score"].mean()}')
# # not split X with splitter = 0.8637825546278203

# # Grid Search
# gd_dt = DecisionTreeClassifier()
# params = {'max_depth':np.arange(5, 30, 1), 'min_impurity_decrease' : np.arange(0.001, 0.01, 0.0001), 'min_samples_split':np.arange(2, 100, 10)}
# grid_cv = GridSearchCV(gd_dt, param_grid = params, cv = splitter)
# grid_cv.fit(X_scaled, y)
# # print(grid_cv.cv_results_.keys())
# '''
# dict_keys(['mean_fit_time', 'std_fit_time', 'mean_score_time', 
# 'std_score_time', 'param_max_depth', 'params', 'split0_test_score', 
# 'split1_test_score', 'split2_test_score', 'split3_test_score', 
# 'split4_test_score', 'mean_test_score', 'std_test_score', 'rank_test_score'])
# '''
# print(grid_cv.best_params_)
# print(grid_cv.best_score_)

# Random Forest
# rf = RandomForestClassifier(n_estimators = 100)
# params = {
#     'n_estimators' : [50, 80, 100, 120],
#     'max_depth' : [3, 5, 7, 9, 12, 15]
# }

# splitter = StratifiedKFold(n_splits = 5, shuffle = True)
# grid_rf = GridSearchCV(rf, param_grid = params, cv = splitter)
# grid_rf.fit(X_scaled, y)
# print(grid_rf.best_params_)
# print(grid_rf.cv_results_['mean_test_score'])
# print(grid_rf.best_score_)
'''
{'max_depth': 15, 'n_estimators': 100}
0.8971806715224728
'''

# rf_best = RandomForestClassifier(n_estimators = 100, max_depth = 15)
# rf_best.fit(X_scaled, y)
# print(rf_best.feature_importances_)
# print(rf_best.score(X_scaled, y))

# GBM
# gbm = GradientBoostingClassifier()
# params = {
#     'n_estimators' : [50, 80, 100, 120],
#     'max_depth' : [3, 5, 7, 9, 12, 15]
# }

# splitter = StratifiedKFold(n_splits = 5, shuffle = True)
# grid_gbm = GridSearchCV(gbm, param_grid = params, cv = splitter)
# grid_gbm.fit(X_scaled, y)
# print(grid_gbm.best_params_)
# print(grid_gbm.cv_results_['mean_test_score'])
# print(grid_gbm.best_score_)
'''
{'max_depth': 7, 'n_estimators': 80}
[0.86609309 0.86547711 0.86593936 0.86917214 0.8742499  0.87332777
 0.87394374 0.87579037 0.88287014 0.88733428 0.88702635 0.88702588
 0.88363972 0.88733381 0.88564268 0.88717996 0.87779073 0.87963664
 0.8802525  0.88071594 0.87055486 0.87332694 0.87163345 0.87455723]
0.8873342808077218
'''


# gbm = GradientBoostingClassifier()
# splitter = StratifiedKFold(n_splits = 5, shuffle = True)
# cv_gbm = cross_validate(gbm, X_scaled, y, cv = splitter)
# print(np.mean(cv_gbm['test_score']))

# titanic = sns.load_dataset('titanic')
# print(titanic.head())
'''
   survived  pclass     sex   age  sibsp  parch     fare  ...  class    who adult_male  deck  embark_town alive  alone
0         0       3    male  22.0      1      0   7.2500  ...  Third    man       True   NaN  Southampton    no  False
1         1       1  female  38.0      1      0  71.2833  ...  First  woman      False     C    Cherbourg   yes  False
2         1       3  female  26.0      0      0   7.9250  ...  Third  woman      False   NaN  Southampton   yes   True
3         1       1  female  35.0      1      0  53.1000  ...  First  woman      False     C  Southampton   yes  False
4         0       3    male  35.0      0      0   8.0500  ...  Third    man       True   NaN  Southampton    no   True

[5 rows x 15 columns]
'''
# print(titanic.info())
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 15 columns):
 #   Column       Non-Null Count  Dtype
---  ------       --------------  -----
 0   survived     891 non-null    int64
 1   pclass       891 non-null    int64
 2   sex          891 non-null    object
 3   age          714 non-null    float64
 4   sibsp        891 non-null    int64
 5   parch        891 non-null    int64
 6   fare         891 non-null    float64
 7   embarked     889 non-null    object
 8   class        891 non-null    category
 9   who          891 non-null    object
 10  adult_male   891 non-null    bool
 11  deck         203 non-null    category
 12  embark_town  889 non-null    object
 13  alive        891 non-null    object
 14  alone        891 non-null    bool
dtypes: bool(2), category(2), float64(2), int64(4), object(5)
memory usage: 80.7+ KB
None
'''
# DT, RF, GB 를 사용하여 survived 예측.

# drop missing value
# titanic.dropna(axis = 0,  subset = ['age'], inplace = True)

# # translate male / female -> True / False
# X = titanic.loc[:, 'pclass':'fare']
# y = titanic['survived']
# # print(f'X = {X}')
# X = pd.get_dummies(X, columns = ['sex'], drop_first = True) # col name is changed... sex -> sex_male
# print(f'update dummy = {X}')
# print(X.info())
''' check missing value
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 6 columns):
 #   Column    Non-Null Count  Dtype
---  ------    --------------  -----
 0   pclass    891 non-null    int64
 1   age       714 non-null    float64
 2   sibsp     891 non-null    int64
 3   parch     891 non-null    int64
 4   fare      891 non-null    float64
 5   sex_male  891 non-null    bool
dtypes: bool(1), float64(2), int64(3)
memory usage: 35.8 KB
None
'''

# print(X.info())
'''
<class 'pandas.core.frame.DataFrame'>
Index: 714 entries, 0 to 890
Data columns (total 6 columns):
 #   Column    Non-Null Count  Dtype
---  ------    --------------  -----
 0   pclass    714 non-null    int64
 1   age       714 non-null    float64
 2   sibsp     714 non-null    int64
 3   parch     714 non-null    int64
 4   fare      714 non-null    float64
 5   sex_male  714 non-null    bool
dtypes: bool(1), float64(2), int64(3)
memory usage: 34.2 KB
None
'''
# print(X['sex_male'].value_counts())
'''
sex_male
True     453
False    261
'''

# scaling
# ss = StandardScaler()
# X_scaled = ss.fit_transform(X)

# Grid Search DT
# dt = DecisionTreeClassifier()
# params_dt = {
#     'max_depth':np.arange(5, 30, 1), 
#     'max_leaf_nodes' : np.arange(1, 15, 1)
# }

# splitter = StratifiedKFold(n_splits = 7, shuffle = True)
# grid_dt = GridSearchCV(dt, param_grid = params_dt, cv = splitter)
# grid_dt.fit(X_scaled, y)
# print(f'best params of DT = {grid_dt.best_params_}')
# print(f'result of DT = {grid_dt.cv_results_["mean_test_score"]}')
# print(f'best score of DT = {grid_dt.best_score_}')

# Grid Search RF
# rf = RandomForestClassifier()
# params_rf = {
#     'n_estimators' : np.arange(50, 200, 10),
#     'max_depth':np.arange(5, 30, 1) 
#     # 'max_leaf_nodes' : np.arange(1, 15, 1)
# }

# splitter = StratifiedKFold(n_splits = 5, shuffle = True)
# grid_rf = GridSearchCV(rf, param_grid = params_rf, cv = splitter)
# grid_rf.fit(X, y)
# print(f'best params of RF = {grid_rf.best_params_}')
# print(f'result of RF = {grid_rf.cv_results_["mean_test_score"]}')
# print(f'best score of RF = {grid_rf.best_score_}')

# Grid Search GB
# gb = GradientBoostingClassifier()
# params_gb = {
#     'learning_rate': np.arange(0.01, 0.03, 0.001),
#     'max_depth':np.arange(5, 30, 1)
#     # 'max_leaf_nodes' : np.arange(1, 15, 1)
# }

# splitter = StratifiedKFold(n_splits = 5, shuffle = True)
# grid_gb = GridSearchCV(gb, param_grid = params_gb, cv = splitter)
# grid_gb.fit(X, y)
# print(f'best params of GB = {grid_gb.best_params_}')
# print(f'result of GB = {grid_gb.cv_results_["mean_test_score"]}')
# print(f'best score of GB = {grid_gb.best_score_}')

# Logistic Regression
wine = pd.read_csv('https://bit.ly/wine-date')
print(wine.head())

wine['class'] = wine['class'].astype('int32').astype('category')
print(wine.head())

X = wine.iloc[:, :-1]
y = wine['class']

ss = StandardScaler()
X_scaled = ss.fit_transform(X)
print(X_scaled[:10])

lr = LogisticRegression(max_iter = 100)
splitter = StratifiedKFold(n_splits = 5, shuffle = True)
scores = cross_validate(lr, X_scaled, y, cv = splitter)

print(scores['test_score'])

lr.fit(X_scaled, y)
print(f'Probability = \n {lr.predict_proba(X_scaled[:10])}')
print(f'coef, intercept = \n {lr.coef_, lr.intercept_}')
print(f'classes = \n {lr.classes_}')
