#!/usr/bin/python3

"""
    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

features_train = features_train[:int(len(features_train)/100)]
labels_train = labels_train[:int(len(labels_train)/100)]

#gamma defines how far the influence of a single training example reaches
#how much regularization is applied to the kernel is inversely proportional to the C parameter
#i.e. controls tradeoff between smooth decision boundary and classifying training points correctly

t0 = time()

clf = SVC(kernel="linear")

clf.fit(features_train, labels_train)

print("Training Time:", round(time()-t0, 3), "s")
t0 = time()

pred = clf.predict(features_test)
print("Predicting Time:", round(time()-t0, 3), "s")
print(f"Test accuracy for Linear SVC: {clf.score(features_test, labels_test)}")
print()
###########################################################################################################
t0 = time()

clf = SVC(kernel="poly", gamma='scale', C=5.0)

clf.fit(features_train, labels_train)
print("Training Time:", round(time()-t0, 3), "s")
t0 = time()

pred = clf.predict(features_test)
print("Predicting Time:", round(time()-t0, 3), "s")
print(f"Test accuracy for Poly SVC: {clf.score(features_test, labels_test)}")
print()

###########################################################################################################
t0 = time()

clf = SVC(kernel="rbf", gamma='scale', C=5.0)

clf.fit(features_train, labels_train)
print("Training Time:", round(time()-t0, 3), "s")
t0 = time()

pred = clf.predict(features_test)
print("Predicting Time:", round(time()-t0, 3), "s")
print(f"Test accuracy for RBF SVC: {clf.score(features_test, labels_test)}")
print()
###########################################################################################################

t0 = time()

clf = SVC(kernel="sigmoid", gamma='scale', C=5.0)

clf.fit(features_train, labels_train)
print("Training Time:", round(time()-t0, 3), "s")
t0 = time()

pred = clf.predict(features_test)
print("Predicting Time:", round(time()-t0, 3), "s")
print(f"Test accuracy for Sigmoid SVC: {clf.score(features_test, labels_test)}")
print()
