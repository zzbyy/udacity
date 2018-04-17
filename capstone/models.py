# http://scikit-learn.org/stable/auto_examples/text/document_classification_20newsgroups.html#sphx-glr-auto-examples-text-document-classification-20newsgroups-py

import numpy as np
import matplotlib.pyplot as plt
import logging
from time import time

from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.utils.extmath import density
from sklearn.feature_selection import SelectFromModel


# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


def benchmark(clf, X_train, y_train, X_test, y_test):
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    score = accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % score)

    if hasattr(clf, 'coef_'):
        print("dimensionality: %d" % clf.coef_.shape[1])
        print("density: %f" % density(clf.coef_))
        print()

    print()
    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time, test_time


def evaluate_models(X_train, y_train, X_test, y_test):
    results = []

    # print('=' * 80)
    # print('Random Forest')
    # rf_clf = RandomForestClassifier(n_estimators=100)
    # results.append(benchmark(rf_clf, X_train, y_train, X_test, y_test))

    for penalty in ['l1', 'l2']:
        print('=' * 80)
        print('%s penalty' % penalty.upper())
        svc = LinearSVC(penalty=penalty, dual=False, tol=0.001)
        results.append(benchmark(svc, X_train, y_train, X_test, y_test))

        sgd = SGDClassifier(alpha=0.0001, max_iter=50, penalty=penalty)
        results.append(benchmark(sgd, X_train, y_train, X_test, y_test))

    # Train SGD with Elastic Net penalty
    print('=' * 80)
    print("Elastic-Net penalty")
    sgd = SGDClassifier(alpha=0.0001, max_iter=50, penalty='elasticnet')
    results.append(benchmark(sgd, X_train, y_train, X_test, y_test))

    # Train sparse Naive Bayes classifiers
    print('=' * 80)
    print("Naive Bayes")
    results.append(benchmark(MultinomialNB(alpha=.01),
                             X_train, y_train, X_test, y_test))
    results.append(benchmark(BernoulliNB(alpha=.01),
                             X_train, y_train, X_test, y_test))

    print('=' * 80)
    print("LinearSVC with L1-based feature selection")
    # The smaller C, the stronger the regularization.
    # The more regularization, the more sparsity.
    results.append(benchmark(Pipeline([
        ('feature_selection', SelectFromModel(
            LinearSVC(penalty="l1", dual=False, tol=1e-3))),
        ('classification', LinearSVC(penalty="l2"))]), X_train, y_train, X_test, y_test))

    # make some plots

    indices = np.arange(len(results))

    results = [[x[i] for x in results] for i in range(4)]

    clf_names, score, training_time, test_time = results
    training_time = np.array(training_time) / np.max(training_time)
    test_time = np.array(test_time) / np.max(test_time)

    plt.figure(figsize=(12, 8))
    plt.title("Score")
    plt.barh(indices, score, .2, label="score", color='navy')
    plt.barh(indices + .3, training_time, .2,
             label="training time", color='c')
    plt.barh(indices + .6, test_time, .2,
             label="test time", color='darkorange')
    plt.yticks(())
    plt.legend(loc='best')
    plt.subplots_adjust(left=.25)
    plt.subplots_adjust(top=.95)
    plt.subplots_adjust(bottom=.05)

    for i, c in zip(indices, clf_names):
        plt.text(-.3, i, c)

    plt.show()
