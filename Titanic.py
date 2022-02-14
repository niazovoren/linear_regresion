import sklearn
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Train Data\titanic_train.csv")
# print(df.head())

# print(df.dtypes)


# Divide the Data to X train and labels.
X_training_data = df.iloc[:, 2:12].copy()
Y_training_label = df.iloc[:, 1:2].copy()
# print(Y_training_label.dtypes)
# print(X_training_data.dtypes)


# Finding unique values in label.
# print(Y_training_label['Survived'].unique())


# Finding Nan values in Data
# print(X_training_data.isnull().sum())
age_median_value = X_training_data['Age'].median(axis=0, skipna=True)
# X_training_data.loc[X_training_data['Age'] == 'NaN'] = age_median_value
X_training_data['Age'] = X_training_data['Age'].fillna(age_median_value)
X_training_data = X_training_data.drop('Cabin', axis=1)
X_training_data = X_training_data.drop('Embarked', axis=1)
# print(X_training_data.isnull().sum())


plt.figure(figsize=(12, 10))
cor = X_training_data.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()
