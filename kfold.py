import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random import shuffle
from preprocess import get_train_data, get_test_data
from dtree import DecisionTree

x_train, y_train = get_train_data()
x_test, y_test = get_test_data()
n_attr = np.size(x_train, axis=1)


# %%
def k_fold_cross_validation(x, y, k, shf=False):
    if shf:
        to_shf = np.column_stack((x, y))
        to_shf = list(to_shf)
        shuffle(to_shf)
        to_shf = np.array(to_shf)
        x = np.delete(to_shf, -1, axis=1)
        y = to_shf[:, -1]
    train_acc = np.zeros((k, n_attr))
    val_acc = np.zeros((k, n_attr))
    for d in range(k):
        print(d, "th fold...")
        x_train = np.array([row for i, row in enumerate(x) if i % k != d])
        x_val = np.array([row for i, row in enumerate(x) if i % k == d])
        y_train = np.array([val for i, val in enumerate(y) if i % k != d])
        y_val = np.array([val for i, val in enumerate(y) if i % k == d])
        for depth in range(n_attr):
            dtree = DecisionTree(x_train, y_train, max_depth=depth)
            dtree.fit()
            # train_acc[d, depth] = dtree.accuracy(x_train, y_train)
            val_acc[d, depth] = dtree.accuracy(x_val, y_val)
    return val_acc


acc = k_fold_cross_validation(x_train, y_train, 5, False)
# %%
acc_df = pd.DataFrame({'Depth': range(1, n_attr + 1),
                       '1st fold': acc[0, :],
                       '2nd fold': acc[1, :],
                       '3rd fold': acc[2, :],
                       '4th fold': acc[3, :],
                       '5th fold': acc[4, :],
                       })
acc_df['mean'] = acc_df.iloc[:, 1:].mean(axis=1)


# %%
def plot_acc(df):
    # plt.plot('Depth', '1st fold', data=df)
    # plt.plot('Depth', '2nd fold', data=df)
    # plt.plot('Depth', '3rd fold', data=df)
    # plt.plot('Depth', '4th fold', data=df)
    # plt.plot('Depth', '5th fold', data=df)
    # plt.legend()
    # plt.xlabel('Depth')
    # plt.savefig('res/kfold_crossval_all.png')
    # plt.show()
    plt.plot('Depth', 'mean', data=df)
    plt.xlabel('Depth')
    plt.ylabel('Mean accuracy')
    plt.show()
    # plt.savefig('res/kfold_crossval_mean.png')
