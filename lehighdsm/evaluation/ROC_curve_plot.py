
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

def ROC_curve_plot(experiment):

    y_test_probabilities = experiment['y_test_probabilities']
    y_train_probabilities = experiment['y_train_probabilities']
    y_test = experiment['y_test']
    y_train = experiment['y_train']

    # fpr1, tpr1, thr1 = roc_curve(y_train, y_train_prediction, pos_label=1)
    # fpr2, tpr2, thr2 = roc_curve(y_test, y_test_prediction, pos_label=1)

    fpr1, tpr1, thr1 = roc_curve(y_train, y_train_probabilities[:, 1], pos_label=1)
    fpr2, tpr2, thr2 = roc_curve(y_test, y_test_probabilities[:, 1], pos_label=1)
    auc1 = roc_auc_score(y_train, y_train_probabilities[:, 1])
    auc2 = roc_auc_score(y_test, y_test_probabilities[:, 1])

    # plot ROC curve and report AUC
    plt.plot(fpr2, tpr2, color='darkorange',
             lw=2, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    experiment['ROC_train'] = (fpr1, tpr1, thr1, auc1)
    experiment['ROC_test'] = (fpr2, tpr2, thr2, auc2)

    # tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()


    return