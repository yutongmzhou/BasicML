#!/usr/bin/python
"""
K-nearest neighbors (KNN) is a simple, non-parametric, supervised machine learning algorithm used for
classification and regression tasks. It operates by storing all available cases and classifying new cases
based on a similarity measure, typically distance metrics like Euclidean distance. For classification, the
algorithm assigns a class to a data point by majority vote of its K nearest neighbors, where K is a user-defined constant.
In regression, it predicts the value based on the average of the values of its K nearest neighbors. KNN is intuitive,
easy to implement, and can handle multi-class classification,
but it can be computationally expensive and sensitive to the choice of K and the distance metric used.
"""

import sys
sys.path.append("../tools/")

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture
from time import time
from sklearn import neighbors

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

num_neighbors = 3

print(f"Number of features: {len(features_train[0])}")
clf = neighbors.KNeighborsClassifier(num_neighbors)
clf.fit(features_train, labels_train)

t0 = time()

print("Training Time:", round(time()-t0, 3), "s")
t0 = time()

pred = clf.predict(features_test)
print("Predicting Time:", round(time()-t0, 3), "s")
print(f"Test accuracy for k = {num_neighbors}: {clf.score(features_test, labels_test)}")
print()


try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass
