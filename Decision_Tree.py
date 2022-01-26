import numpy as np
import pandas as pd


class DecisionTree:

    def __init__(self, X_Train_data, y_train_label):
        self.x_train = X_Train_data
        self.y_train = y_train_label
        self.calculated_mean = self.Calc_mean()

    def Calc_mean(self):
        self.calculated_mean = np.zeros((self.x_train.shape[0], self.x_train.shape[1]))
        x_train_copy = self.x_train.copy()
        np.sort(x_train_copy, axis=0)
        for i in range(self.x_train.shape[1]):
            for j in range(self.x_train.shape[0]):
                self.calculated_mean[j, i] = np.average(x_train_copy[j:j + 2, i])

        return self.calculated_mean

    def gini_calc(self, Column_name, label):
        edge_smaller_then_SV = self.x_train[Column_name] < self.calculated_mean[Column_name]
        edge_smaller_then_SV_yes = self.y_train[edge_smaller_then_SV] == 'yes'
        len(edge_smaller_then_SV_yes)

        return

    def Split_value(self):
        pass

    def Find_Root(self):
        pass

    def fit(self):
        pass


df = pd.read_csv(r'C:\Users\orenn\OneDrive - Kornit Digital\Desktop\wdbc.data')
# #  Checking if the data is OK.

# print(df.dtypes)
# print(df['M'].unique())
# print(len(df.columns))
# print(len(df))
X_Data = df.iloc[0:568, 2:12]
X_Data.columns = ['Radius', 'Texture', 'Perimeter', 'Area', 'Smoothness', 'Compactness', 'Concavity', 'Concave Points',
                  'Symmetry', 'Fractal Dimension']
# print(X_Data.head())
y_label = df.iloc[0:568, 1]


## show to OR
# import numpy as np
#
# label = np.array(['yes', 'yes', 'No', 'yes'])
# for i in range(len(label)):
#     d = label[i:len(label)+1]
#     u, indices = np.unique(d, return_counts=True)
#     print(u, indices)