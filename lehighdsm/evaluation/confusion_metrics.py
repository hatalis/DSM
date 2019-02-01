
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import itertools

def confusion_metrics(experiment, plot_matrix = False):
    """
    Computes performance metrics for trained ML model.

    Args:
        experiment(dict): Dictionary containing trained ML model results
        plot_matrix(bool): plot confusion matrix

    Returns:
        experiment(dict): Experiment dictionary with additional or updated keys.
    """

    # y_train = experiment['y_train']
    # y_train_prediction = experiment['y_train_prediction']
    y_test = experiment['y_test']
    y_test_prediction = experiment['y_test_prediction']

    # precision1, recall1, fbeta1, support1 = precision_recall_fscore_support(y_train, y_train_prediction)
    precision2, recall2, fbeta2, support2 = precision_recall_fscore_support(y_test, y_test_prediction)

    # accuracy1 = accuracy_score(y_train, y_train_prediction)
    accuracy2 = accuracy_score(y_test, y_test_prediction)

    # TPR and TNR (TPR should equal to recall)
    # TPR1 = np.mean(y_train[y_train_prediction == 1])
    # NPR1 = 1. - np.mean(y_train[y_train_prediction == 0])
    TPR2 = np.mean(y_test[y_test_prediction == 1])
    NPR2 = 1. - np.mean(y_test[y_test_prediction == 0])

    # experiment['SCORES_train'] = (precision1, recall1, fbeta1, support1, accuracy1, TPR1, NPR1)
    experiment['SCORES_test'] = (precision2, recall2, fbeta2, support2, accuracy2, TPR2, NPR2)

    print('')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('Testing Results:')
    print('   Class 0')
    print('      Precision = {:,.2f}%'.format(precision2[0] * 100))
    print('      Recall    = {:,.2f}%'.format(recall2[0] * 100))
    print('      F1-Score  = {:,.2f}%'.format(fbeta2[0] * 100))
    print('      Support   = {:,.0f}'.format(support2[0]))
    print('   Class 1')
    print('      Precision = {:,.2f}%'.format(precision2[1] * 100))
    print('      Recall    = {:,.2f}%'.format(recall2[1] * 100))
    print('      F1-Score  = {:,.2f}%'.format(fbeta2[1] * 100))
    print('      Support   = {:,.0f}'.format(support2[1]))
    print('True positive rate   = {:,.2f}'.format(TPR2 * 100))
    print('True negative rate   = {:,.2f}'.format(NPR2 * 100))
    print('Accuracy    = {:,.2f}%'.format(accuracy2 * 100))
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('')

    if plot_matrix:
        cnf_matrix = confusion_matrix(y_test, y_test_prediction)
        plot_confusion_matrix(cnf_matrix, ['$H_0$', '$H_1$'])

    return experiment


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()