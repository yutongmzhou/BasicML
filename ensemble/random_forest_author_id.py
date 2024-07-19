#!/usr/bin/python

"""
    Random Forest is an ensemble learning algorithm used for classification and regression tasks that builds multiple decision
    trees and merges their results to produce a more accurate and stable prediction. It operates by constructing a multitude of
    decision trees during training and outputs either the mode of the classes (classification) or the mean prediction (regression)
    of the individual trees. Each tree in the forest is trained on a random subset of the data and a random subset of features, which
    helps to reduce overfitting and improve generalization. The diversity among the trees, achieved through these random selections,
    allows the model to be robust and handle large datasets with higher dimensionality. Random Forest is known for its high accuracy,
    ability to handle missing values, and resistance to overfitting, though it can be computationally intensive and less interpretable
    than single decision trees.
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
clf = ensemble.RandomForestClassifier(n_estimators=num_estimators)
clf.fit(features_train, labels_train)

t0 = time()

print("Training Time:", round(time()-t0, 3), "s")
t0 = time()

pred = clf.predict(features_test)
print("Predicting Time:", round(time()-t0, 3), "s")
print(f"Test accuracy for Random Forest num_estimators = {num_estimators}: {clf.score(features_test, labels_test)}")
print()


try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass
