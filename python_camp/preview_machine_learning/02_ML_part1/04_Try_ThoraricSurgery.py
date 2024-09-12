import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler


thoraric = pd.read_csv('../01_Data_handling/dataset/ThoraricSurgery.csv', header = None)
# print(thoraric.head())
'''
    0   1     2     3   4   5   6   7   8   9   10  11  12  13  14  15  16  17
0  293   1  3.80  2.80   0   0   0   0   0   0  12   0   0   0   1   0  62   0
1    1   2  2.88  2.16   1   0   0   0   1   1  14   0   0   0   1   0  60   0
2    8   2  3.19  2.50   1   0   0   0   1   0  11   0   0   1   1   0  66   1
3   14   2  3.98  3.06   2   0   0   0   1   1  14   0   0   0   1   0  80   1
4   17   2  2.21  1.88   0   0   1   0   0   0  12   0   0   0   1   0  56   0
'''
# print(thoraric.info())
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 470 entries, 0 to 469
Data columns (total 18 columns):
 #   Column  Non-Null Count  Dtype
---  ------  --------------  -----
 0   0       470 non-null    int64
 1   1       470 non-null    int64
 2   2       470 non-null    float64
 3   3       470 non-null    float64
 4   4       470 non-null    int64
 5   5       470 non-null    int64
 6   6       470 non-null    int64
 7   7       470 non-null    int64
 8   8       470 non-null    int64
 9   9       470 non-null    int64
 10  10      470 non-null    int64
 11  11      470 non-null    int64
 12  12      470 non-null    int64
 13  13      470 non-null    int64
 14  14      470 non-null    int64
 15  15      470 non-null    int64
 16  16      470 non-null    int64
 17  17      470 non-null    int64
dtypes: float64(2), int64(16)
memory usage: 66.2 KB
None
'''

# change col name.. 0~17 -> variable name..
col_name = ['id', 'DGN', 'PRE4', 'PRE5', 'PRE6', 
            'PRE7', 'PRE8', 'PRE9', 'PRE10', 'PRE11', 
            'PRE14', 'PRE17', 'PRE19', 'PRE25', 
            'PRE30', 'PRE32', 'AGE', 'Risk1Yr']

for idx, col in enumerate(col_name):
    thoraric.rename(columns = {idx:col}, inplace = True)

# categrical col change dummy.
category_col_name = []
# print(thoraric.head())
'''
    id  DGN  PRE4  PRE5  PRE6  PRE7  PRE8  PRE9  PRE10  PRE11  PRE14  PRE17  PRE19  PRE25  PRE30  PRE32  AGE  Risk1Yr
0  293    1  3.80  2.80     0     0     0     0      0      0     12      0      0      0      1      0   62        0
1    1    2  2.88  2.16     1     0     0     0      1      1     14      0      0      0      1      0   60        0
2    8    2  3.19  2.50     1     0     0     0      1      0     11      0      0      1      1      0   66        1
3   14    2  3.98  3.06     2     0     0     0      1      1     14      0      0      0      1      0   80        1
4   17    2  2.21  1.88     0     0     1     0      0      0     12      0      0      0      1      0   56        0
'''

# choose X / y
X = thoraric.loc[:, 'DGN':'AGE']
y = thoraric['Risk1Yr']
# print(X.shape) (470, 16)
# print(y.shape) (470,)



# Using RF
rf = RandomForestClassifier(n_estimators = 100)
params = {
    'n_estimators' : np.arange(50, 200, 10),
    'max_depth' : np.arange(3, 50, 2)
}

splitter = StratifiedKFold(n_splits = 5, shuffle = True)
grid_rf = GridSearchCV(rf, param_grid = params, cv = splitter)
grid_rf.fit(X, y)
print(grid_rf.best_params_)
print(grid_rf.cv_results_['mean_test_score'])
print(grid_rf.best_score_) 
# result of GridSearch.. not preprocessing
# {'max_depth': 3, 'n_estimators': 50}
# 0.851....


# rf_best = RandomForestClassifier(n_estimators = 100, max_depth = 15)
# rf_best.fit(X_scaled, y)
# print(rf_best.feature_importances_)
# print(rf_best.score(X_scaled, y))


