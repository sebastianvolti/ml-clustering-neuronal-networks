import pandas as pd

from sklearn import neighbors
from sklearn.model_selection import cross_val_score


def scikit_knn(training_examples, training_classes, k=5, **kwargs):
    knn = neighbors.KNeighborsClassifier(n_neighbors=k).fit(training_examples, training_classes.values.ravel())

    return knn


def cross_val(clf, X, y, scoring='accuracy', cv_folds=5):
	return cross_val_score(clf, X, y, scoring=scoring, cv=cv_folds)
