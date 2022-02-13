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
print(X_training_data.isnull().sum())


# plt.figure(figsize=(12, 10))
# cor = df.corr()
# sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
# plt.show()
