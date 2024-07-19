#!/usr/bin/python3

""" 
    Naive Bayes is a simple, probabilistic classifier based on Bayes' Theorem, widely used for classification tasks.
    It operates under the assumption of conditional independence, meaning it presumes that the features contributing to
    the prediction are independent of each other given the class label, which is often not true in real-world scenarios.
    Despite this "naive" assumption, Naive Bayes classifiers perform remarkably well in many practical applications,
    particularly with large datasets. They calculate the probability of each class given a set of features and predict the
    class with the highest posterior probability. Naive Bayes is efficient, easy to implement, and works well with both continuous
    and discrete data, making it suitable for text classification tasks like spam detection and sentiment analysis. However, its
    performance can be suboptimal when features are highly correlated.
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

clf = GaussianNB()

t0 = time()
clf.fit(features_train, labels_train)

print("Training Time:", round(time()-t0, 3), "s")

t0 = time()

predictions = clf.predict(features_test)


accuracy = accuracy_score(predictions, labels_test)
print(f"Test accuracy: {clf.score(features_test, labels_test)}")

print("Predicting Time:", round(time()-t0, 3), "s")
