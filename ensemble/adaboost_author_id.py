#!/usr/bin/python

"""
    AdaBoost, short for Adaptive Boosting, is an ensemble learning algorithm designed to improve the performance of weak classifiers
     by combining them into a strong classifier. The process involves training multiple weak classifiers, typically decision stumps
     (simple decision trees with a single split), in a sequential manner. Each classifier is trained on the same dataset, but with adjusted
     weights for the training instances based on the performance of the previous classifiers. Misclassified instances are given higher weights
     so that subsequent classifiers focus more on these difficult cases. The final model is a weighted sum of all the weak classifiers, where
     the weights are determined by their respective accuracies. AdaBoost is effective in reducing bias and variance, making it robust against
     overfitting, and is particularly known for its high accuracy and simplicity. However, it can be sensitive to noisy data and outliers.
"""


import sys
sys.path.append("../tools/")

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture
from time import time
from sklearn import ensemble

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
################################################################################

t0 = time()

num_estimators = 100

print(f"Number of features: {len(features_train[0])}")
clf = ensemble.AdaBoostClassifier(n_estimators=num_estimators)
clf.fit(features_train, labels_train)

t0 = time()

print("Training Time:", round(time()-t0, 3), "s")
t0 = time()

pred = clf.predict(features_test)
print("Predicting Time:", round(time()-t0, 3), "s")
print(f"Test accuracy for Adaboost num_estimators = {num_estimators}: {clf.score(features_test, labels_test)}")
print()


try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass
