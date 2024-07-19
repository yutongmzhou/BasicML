#!/usr/bin/python

"""
    Use a Decision Tree to identify emails from the Enron corpus by author:
    Sara has label 0
    Chris has label 1
"""

"""
    Decision trees are a type of supervised machine learning algorithm used for classification and regression tasks. 
    They work by splitting the data into subsets based on the value of input features, creating a tree-like model of decisions. 
    Each internal node of the tree represents a feature or attribute, each branch represents a decision rule, and each leaf node 
    represents an outcome or class label. The tree is constructed through a process called recursive partitioning, where the algorithm 
    selects the best feature to split the data at each node based on criteria such as Gini impurity or information gain for classification, 
    or mean squared error for regression. Decision trees are easy to understand and interpret, can handle both numerical and categorical data, 
    and require little data preprocessing. However, they can be prone to overfitting and may not generalize well to unseen data unless properly 
    pruned or used in ensemble methods like random forests.
"""

import sys
from time import time

sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn import tree

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

print(f"Number of features: {len(features_train[0])}")
clf = tree.DecisionTreeClassifier(min_samples_split=40)
clf.fit(features_train, labels_train)

t0 = time()

print("Training Time:", round(time()-t0, 3), "s")
t0 = time()

pred = clf.predict(features_test)
print("Predicting Time:", round(time()-t0, 3), "s")
print(f"Test accuracy for Decision Tree with min_sample_split=40: {clf.score(features_test, labels_test)}")
print()

