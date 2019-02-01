
import matplotlib.pyplot as plt

def plot_ROC_curve(FPR, TPR, label,lw=1):
    plt.plot(FPR, TPR, color='darkorange',
             lw=lw, label=label)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    # plt.grid(linestyle='--')
    return None