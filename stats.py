# %%
import numpy as np
from preprocess import get_train_data, get_test_data
from dtree import DecisionTree

x_train, y_train = get_train_data()
x_test, y_test = get_test_data()
decision_tree = DecisionTree(x_train, y_train, max_depth=1)
decision_tree.fit()
decision_tree.traverse()
y_hat = decision_tree.predict(x_test)
print("accuracy: ", decision_tree.accuracy(x_test, y_test))


# %%
def get_stats():
    TP = np.sum(np.logical_and(y_test == 1, y_hat == 1))
    FP = np.sum(np.logical_and(y_test == 0, y_hat == 1))
    TN = np.sum(np.logical_and(y_test == 0, y_hat == 0))
    FN = np.sum(np.logical_and(y_test == 1, y_hat == 0))
    return TP, FP, TN, FN


def specificity():
    TP, FP, TN, FN = get_stats()
    return TN / (TN + FP)


def sensitivity():
    TP, FP, TN, FN = get_stats()
    return TP / (TP + FN)


# %%
spc = specificity()
sen = sensitivity()
print("specificity: ", spc)
print("sensitivity: ", sen)
