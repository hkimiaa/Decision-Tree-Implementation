import pandas as pd
import numpy as np
from dtree import DecisionTree

train_data = pd.read_csv('Dataset/train.txt', sep='\t')
test_data = pd.read_csv('Dataset/test.txt', sep='\t')


# %%
def clean_data(data):
    data = pd.DataFrame.copy(data)
    data['Race'] = data['Race'].str.capitalize()
    data['Gender'].replace({'F': 'FM', 'female': 'FM', 'male': 'M'}, inplace=True)
    data.replace({
        'False': 'F',
        'True': 'T',
        'false': 'F',
        'true': 'T',
        '0': 'F',
        '1': 'T'
    }, inplace=True)
    data.replace({'F': 0, 'T': 1}, inplace=True)
    return data


# %%

def get_train_data():
    train_data_clean = clean_data(train_data)
    train_mat = train_data_clean.drop(['tag'], axis=1).to_numpy()
    train_labels = train_data_clean.tag.to_numpy().astype(int)
    return train_mat, train_labels


def get_test_data():
    test_data_clean = clean_data(test_data)
    test_mat = test_data_clean.drop(['tag'], axis=1).to_numpy()
    test_labels = test_data_clean.tag.to_numpy().astype(int)
    return test_mat, test_labels


# %%
# x_train, y_train = get_train_data()
# x_test, y_test = get_test_data()
# decision_tree = DecisionTree(x_train, y_train, max_depth=3)
# decision_tree.fit()
# decision_tree.traverse()

# %%
# train_accuracy = decision_tree.accuracy(train_mat, train_labels)
# test_accuracy = decision_tree.accuracy(test_mat, test_labels)
#
# print("train accuracy: ", train_accuracy)
# print("test accuracy: ", test_accuracy)
