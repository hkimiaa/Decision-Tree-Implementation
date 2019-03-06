import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from preprocess import get_train_data, get_test_data
from dtree import DecisionTree

x_train, y_train = get_train_data()
x_test, y_test = get_test_data()
n_attr = np.size(x_train, axis=1)


def learn_depths():  # training decision tree for different heights
    train_acc = np.zeros(n_attr)
    test_acc = np.zeros(n_attr)
    for depth in range(n_attr):
        dtree = DecisionTree(x_train, y_train, max_depth=depth)
        dtree.fit()
        train_acc[depth] = dtree.accuracy(x_train, y_train)
        test_acc[depth] = dtree.accuracy(x_test, y_test)
    df = pd.DataFrame({'depth': range(1, n_attr + 1), 'Train accuracy': train_acc, 'Test accuracy': test_acc})
    # df.to_csv('res/acc.csv')
    return df


def plot_acc(df):
    plt.plot('depth', 'Train accuracy', data=df)
    plt.plot('depth', 'Test accuracy', data=df)
    plt.legend()
    plt.xlabel('Depth')
    plt.show()
    # plt.savefig('res/acc_vs_depth.png')
