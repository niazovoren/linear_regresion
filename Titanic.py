from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Train Data\titanic_train.csv")
df2 = pd.read_csv(r"C:\Train Data\titanic_test.csv")
# print(df.head())

# print(df.dtypes)


# # Divide the Data to X train and labels.
X_training_data = df.iloc[:, 2:12].copy()
Y_training_label = df.iloc[:, 1:2].copy()
# # print(Y_training_label.dtypes)
# # print(X_training_data.dtypes)


# Divide Test Data
X_test_data = df.iloc[:, 2:12].copy()
Y_test_label = df.iloc[:, 1:2].copy()
age_median_value = X_test_data['Age'].median(axis=0, skipna=True)
X_test_data['Age'] = X_test_data['Age'].fillna(age_median_value)
X_test_data = X_test_data.drop('Cabin', axis=1)
X_test_data = X_test_data.drop('Embarked', axis=1)
X_test_data = X_test_data.drop('Name', axis=1)
X_test_data = X_test_data.drop('Ticket', axis=1)
X_test_data['Sex'].replace(['female', 'male'], [0, 1], inplace=True)
X_test_array = X_test_data.to_numpy()
Y_test_array = Y_test_label.to_numpy().ravel()

# # Finding unique values in Train label.
# # print(Y_training_label['Survived'].unique())


# # Finding Nan values in Train Data
# print(X_training_data.isnull().sum())
age_median_value = X_training_data['Age'].median(axis=0, skipna=True)
# # X_training_data.loc[X_training_data['Age'] == 'NaN'] = age_median_value
X_training_data['Age'] = X_training_data['Age'].fillna(age_median_value)
X_training_data = X_training_data.drop('Cabin', axis=1)
X_training_data = X_training_data.drop('Embarked', axis=1)
X_training_data = X_training_data.drop('Name', axis=1)
X_training_data = X_training_data.drop('Ticket', axis=1)
# X_training_data = X_training_data.drop('Sex', axis=1)
# X_training_data = X_training_data.drop('Age', axis=1)
X_training_data['Sex'].replace(['female', 'male'], [0, 1], inplace=True)
# print(X_training_data.dtypes)
# print(X_training_data['Ticket'].unique())
# # print(X_training_data.isnull().sum())
# print(X_training_data.head(10))
# plt.figure(figsize=(12, 10))
# cor = X_training_data.corr()
# sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
# plt.show()

X_train_array = X_training_data.to_numpy()
Y_training_array = Y_training_label.to_numpy().ravel()
# print(Y_training_array.shape)
# print(X_train_array.shape)


model = LogisticRegression()
solvers = ['newton-cg','liblinear']
penalty = ['l2']
c_values = [100, 10, 1.0, 0.1, 0.01]
grid = dict(solver=solvers, penalty=penalty, C=c_values)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy', error_score=0)
grid_result = grid_search.fit(X_train_array, Y_training_array)
print(grid_result.best_score_, grid_result.best_params_)
result = grid_result.predict(X_test_array)
print(accuracy_score(Y_test_array, result))
#
#
# model = KNeighborsClassifier()
# n_neighbors = [5, 8, 12]
# weights = ['uniform', 'distance']
# algorithm = ['ball_tree', 'kd_tree', 'brute']
# grid = dict(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm)
# cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy', error_score=0)
# grid_result = grid_search.fit(X_train_array, Y_training_array)
# print(grid_result.best_score_, grid_result.best_params_)
# result = grid_result.predict(X_test_array)
# print(accuracy_score(Y_test_array, result))


# model = BaggingClassifier()
# n_estimators = [10, 100, 1000]
# # define grid search
# grid = dict(n_estimators=n_estimators)
# cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
# grid_result = grid_search.fit(X_train_array, Y_training_array)
# print(grid_result.best_score_, grid_result.best_params_)
# result = grid_result.predict(X_test_array)
# print(accuracy_score(Y_test_array, result))


# model = RandomForestClassifier()
# n_estimators = [10, 100, 1000]
# max_features = ['sqrt', 'log2']
# # define grid search
# grid = dict(n_estimators=n_estimators,max_features=max_features)
# cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
# grid_result = grid_search.fit(X_train_array, Y_training_array)
# print(grid_result.best_score_, grid_result.best_params_)



