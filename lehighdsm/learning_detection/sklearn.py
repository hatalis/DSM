
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix


def sklearn(experiment, method, prediction_threshold=0.5, **kwargs):
    """
    Classifier models for binary prediction, as defined in the sklearn-module.

    Args:
        experiment(dict): Experiment dictionary with train- and test-set and labels
        method(int): 0: k-Nearest Neighbors
                     1: Logistic Regression
                     2: Random Forest
                     3: Support Vector Classifier
                     4: Gaussian Naive Bayes
                     5: Decision Trees
                     6: AdaBoost Classifier
                     7: Gradient Boosting Classifier
                     8: Neural Network Classifier
        prediction_threshold(float): Threshold value for predicted label 1 (0 if predicted value is below)
        kwargs: All further keyword-arguments are passed directly to the classifier for custom settings

    Returns:
        dict: Experiment dictionary with additional keys.
    """
    experiment['method'] = method
    experiment['prediction_threshold'] = prediction_threshold
    X_train = experiment['X_train']
    X_test = experiment['X_test']
    y_train = experiment['y_train']


    classifier = None
    if method == 0:
        # k-Nearest Neighbors
        classifier = KNeighborsClassifier(**kwargs)
    elif method == 1:
        # Logistic Regression
        classifier = LogisticRegression(**kwargs)
    elif method == 2:
        # Random Forest
        classifier = RandomForestClassifier(**kwargs)
    elif method == 3:
        # Support Vector Classifier
        classifier = SVC(kernel = 'rbf')  # kernel = linear, poly, rbf, sigmoid
    elif method == 4:
        # Gaussian Naive Bayes
        classifier = GaussianNB(**kwargs)
    elif method == 5:
        # Decision Trees
        classifier = DecisionTreeClassifier(**kwargs)
    elif method == 6:
        # AdaBoost Classifier
        classifier = AdaBoostClassifier(**kwargs)
    elif method == 7:
        # Gradient Boosting Classifier
        classifier = GradientBoostingClassifier(**kwargs)
    elif method == 8:
        # Neural Network Classifier
        classifier = MLPClassifier(**kwargs)
        # classifier = MLPClassifier(hidden_layer_sizes=(10, 5))
    else:
        print('Invalid method!')

    classifier.fit(X_train, np.ravel(y_train))

    # output probability of prediction, use threshold to pick class
    y_train_probabilities = classifier.predict_proba(X_train)
    y_test_probabilities = classifier.predict_proba(X_test)


    y_test = experiment['y_test']

    FPR, TPR, prediction_threshold = roc_curve(y_test, y_test_probabilities[:, 1], pos_label=1)

    N_roc = np.shape(FPR)[0]
    best_d = 10
    best_i = 0
    d = np.ones((N_roc, 1))
    for i in range(N_roc):
        d[i] = np.sqrt((1 - TPR[i]) ** 2 + FPR[i] ** 2)
        if best_d > d[i]:
            best_d = d[i]
            best_i = i

    threshold = prediction_threshold[best_i]
    # auc2 = roc_auc_score(y_test, y_test_probabilities[:, 1])
    y_train_prediction = (y_train_probabilities[:, 1] >= threshold) * 1
    y_test_prediction = (y_test_probabilities[:, 1] >= threshold) * 1

    experiment['FPR'] = FPR
    experiment['TPR'] = TPR
    experiment['y_test_probabilities'] = y_test_probabilities
    experiment['y_train_probabilities'] = y_train_probabilities
    experiment['y_test_prediction'] = y_test_prediction
    experiment['y_train_prediction'] = y_train_prediction

    return experiment
