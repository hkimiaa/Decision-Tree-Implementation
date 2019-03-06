import collections

import numpy as np


def entropy(labels):
    _, val_freq = np.unique(labels, return_counts=True)
    val_prob = val_freq / len(labels)
    return -val_prob.dot(np.log2(val_prob))


def info_gain(attr_col, labels):
    e_labels = entropy(labels)
    print("parent entropy: ", e_labels)
    e_vals = 0.
    vals, freqs = np.unique(attr_col, return_counts=True)
    for i in range(len(vals)):
        count = freqs[i]
        val = vals[i]
        e_vals += count * entropy(labels[attr_col == val])
        print("val: ", val)
        print("freq: %f\tent:%f" % (count / len(labels), entropy(labels[attr_col == val])))
    e_vals = e_vals / len(labels)
    return e_labels - e_vals


class DecisionNode:

    def __init__(self, depth=0, attribute=-1, label=-1, parent=None):
        self.children = {}
        self.attribute = attribute
        self.label = label
        self.parent = parent
        self.depth = depth
        # print("node made!", sep='\t')
        # self.print_node()

    def add_child(self, value, node):
        self.children[value] = node
        node.parent = self
        # print("child with value %s added to below node: " % value)
        # self.print_node()

    def print_node(self):
        # if self.parent is not None:
        #     print("attr=%d \tlabel=%d \tparent=%d" % (self.attribute, self.label, self.parent))
        # else:
        print("attr=%d \tlabel=%d" % (self.attribute, self.label))


def id3(examples, labels, attributes, target_values, max_depth, leaf_purity):  # add base conditions
    """
    :param leaf_purity > percentage of dominant tag -> node=left
    :param max_depth:
    :param examples: x
    :param labels: y
    :param attributes: left attributes to use as a node in this subtree
    :param target_values: values that each attribute can get (for handling nodes with 0 number of examples)
    :return: root node
    """

    def find_best_attr(func=info_gain):
        max_val = -float('inf')
        final_attr = -1
        for attr in attributes:
            attr_col = examples[:, attr]
            new_val = func(attr_col, labels)
            print("------------------- attr: %d\tIG: %f" % (attr, new_val))
            if new_val > max_val:
                max_val = new_val
                final_attr = attr
        return final_attr

    dominant_label = np.bincount(labels).argmax()
    depth = len(target_values) - np.size(attributes)

    tags, freqs = np.unique(labels, return_counts=True)
    # if len(tags) == 1 or len(attributes) == 0 or depth == max_depth:
    if (np.max(freqs) / len(labels)) >= leaf_purity \
            or len(attributes) == 0 or depth == max_depth:
        return DecisionNode(label=dominant_label)

    best_attr = find_best_attr()
    # print("best attr: ", best_attr)
    root = DecisionNode(attribute=best_attr)

    # attr_values = np.unique(examples[:, best_attr])
    attr_values = target_values[best_attr]
    for val in attr_values:
        ind = examples[:, best_attr] == val
        examples_val = examples[ind]
        labels_val = labels[ind]
        if np.sum(ind) == 0:
            child = DecisionNode(label=dominant_label)
        else:
            child = id3(examples_val, labels_val,
                        np.delete(attributes, np.where(attributes == best_attr)),
                        target_values, max_depth, leaf_purity)
        root.add_child(val, child)
    return root


class DecisionTree:

    def __init__(self, examples, labels, max_depth, purity=1, alg=id3):
        """
        :param examples: training examples
        :param labels: labels of examples
        :param alg: learning algorithm
        """
        self.attributes = np.array(range(np.size(examples, 1)))
        self.x = examples
        self.y = labels
        self.max_depth = max_depth
        self.purity = purity
        self.alg = alg
        self.root = None

    def fit(self):
        target_values = {}
        for attr in self.attributes:
            target_values[attr] = np.unique(self.x[:, attr])
        self.root = self.alg(self.x, self.y, self.attributes, target_values,
                             max_depth=self.max_depth, leaf_purity=self.purity)

    def traverse(self):
        visited, queue = set(), collections.deque([self.root])
        visited.add(self.root)
        while queue:
            v: DecisionNode = queue.popleft()
            for val, u in v.children.items():
                if u.parent is not None:
                    print("val=%s\tattr=%d\tlabel=%d\tparent_attr=%d" % (val, u.attribute, u.label, u.parent.attribute))
                visited.add(u)
                queue.append(u)

    def predict(self, x):
        n = np.size(x, axis=0)
        y_pred = []
        for i in range(n):
            # print(x[i, :])
            v: DecisionNode = self.root
            while v.label == -1:
                value = x[i, v.attribute]
                v = v.children[value]
                # v.print_node()
            # print(v.label)
            y_pred.append(v.label)
        return np.array(y_pred)

    def accuracy(self, x, y):
        y_pred = self.predict(x)
        return np.mean(y_pred == y)
