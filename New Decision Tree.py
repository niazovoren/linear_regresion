import numpy as np
import pandas as pd


class DecisionTree:

    def __init__(self, X_Train_data, y_train_label):
        self.x_train = X_Train_data
        self.y_train = y_train_label

    def Split(self, feature_index, value):
        mask = self.x_train[:, feature_index] >= value
        right_dataset = self.x_train[mask]
        left_dataset = self.x_train[~mask]
        right_labels = self.y_train[mask]
        left_labels = self.y_train[~mask]
        return right_dataset, right_labels, left_dataset, left_labels



