import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve


def plot_recall_fpr(y_true, scores):
    fpr, tpr, _ = roc_curve(y_true, scores)
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("Recall")
    plt.title("Recall vs FPR")
    plt.grid(True)
