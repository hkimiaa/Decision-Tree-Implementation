import numpy as np
from scipy import stats
from preprocess import get_train_data, get_test_data
from dtree import DecisionTree

x_train, y_train = get_train_data()
x_test, y_test = get_test_data()
n_attr = np.size(x_train, axis=1)


def learn_with_purity(purity):  # training decision tree with specific purity
    dtree = DecisionTree(x_train, y_train, max_depth=n_attr, purity=purity)
    dtree.fit()
    # train_accuracy = dtree.accuracy(x_train, y_train)
    # test_accuracy = dtree.accuracy(x_test, y_test)
    test_preds = dtree.predict(x_test)
    return test_preds


normal_y = learn_with_purity(1)
prune_y = learn_with_purity(0.8)


# %% paired t-test on data
def paired_ttest(a, b):
    if np.size(a) != np.size(b):
        return None
    n = np.size(a)
    mu_a = np.mean(a)
    mu_b = np.mean(b)
    a_hat = np.subtract(a, mu_a)
    b_hat = np.subtract(b, mu_b)
    t_stat = (mu_a - mu_b) * np.sqrt(n * (n - 1) / np.sum(np.square(a_hat - b_hat)))
    p_val = stats.t.sf(np.abs(t_stat), n - 1) * 2
    return t_stat, p_val


t_stat, p_val = paired_ttest(normal_y, prune_y)
print("p-value: ", p_val)
