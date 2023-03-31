from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np

def get_top_n_accuracy(Y, Ypred, n=5):
    """return top n accuracy. top n accuracy measures how often the true label is in the top n predicted labelsin the softmax distribution"""
    if(Ypred.shape[0] < n):
        print("Invalid N")
        return
    m = len(Y)
    top_n_indx = np.argsort(Ypred, axis=1)[:, -n:][:, ::-1]
    right = np.sum(np.max(Y == top_n_indx, 1))
    return (right/ m)

def get_precision_metric(y, y_pred):
    """Returns the precision using sklearn library"""
    precision = metrics.precision_score(y, y_pred, average="macro")
    return precision

def get_recall_metric(y, y_pred):
    """Returns the recall using sklearn library"""
    recall = metrics.recall_score(y, y_pred, average="macro")
    return recall


def get_f1_metric(y, y_pred):
    """Returns the F1 using sklearn library"""
    f1 = metrics.f1_score(y, y_pred, average="macro")
    return f1

def get_accuracy_metric(y, y_pred):
    """Returns the accuracy using sklearn library"""
    return metrics.accuracy_score(y, y_pred)

def get_confussion_matrix(y, y_pred, classes=None, plot=False, title=None):
    """
    Returns the the confussion matrix using the true and predicted labels
    Arguments: 
        y: The ground truth labels
        y_pred: The predicted labels
        classes: The name of the classes
        plot: If set to true the confussion matrix will be plotted
        title: Title for the confussion matrix plt 
    """
    confusion_matrix = metrics.confusion_matrix(y, y_pred)
    if plot:
        confussion_matrix_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = classes)
        figure = plt.figure(figsize=(15, 15))
        confussion_matrix_display.plot()
        if title is not None:
            plt.title(title)
        plt.show()
    return confusion_matrix

def get_top_accuracies(y, y_pred, n_range, plot=False):
    """
    Returns top n accuracies
    Arguments: 
        y: The ground truth labels
        y_pred: The predicted labels
        n_range: The maximum n range to get the top n accuracy
        plot: If set to True a bar chart will be plotted for the calculated top n accuracies
    """
    acc_arr = []
    for n in n_range:
        acc_arr.append(get_top_n_accuracy(y, y_pred, n))
    if plot:
        plt.bar(n_range, acc_arr)
        plt.title("Top-n Accuracy Vs n")
        plt.xlabel("n")
        plt.ylabel("Top-n Accuracy")
        plt.show()
    return acc_arr

def print_top_5_accuracies(y, y_pred, title=""):
    """Given the ground truth and precited labels this function prints top 1, top 2 to top 5 accuracy"""
    print(f"{title} top-5 Accuracy: {get_top_n_accuracy(y, y_pred, 5)}")
    print(f"{title} top-4 Accuracy: {get_top_n_accuracy(y, y_pred, 4)}")
    print(f"{title} top-3 Accuracy: {get_top_n_accuracy(y, y_pred, 3)}")
    print(f"{title} top-2 Accuracy: {get_top_n_accuracy(y, y_pred, 2)}")
    print(f"{title} top-1 Accuracy: {get_top_n_accuracy(y, y_pred, 1)}")


def get_evaluation_metrics(y, y_pred, DO_PRINT=False):
    """Returns a summary of popular performance evaluation metrics. specifically, accuracy, precision, recall and F1"""
    metrics = {}
    metrics["accuracy"] = get_accuracy_metric(y, y_pred)
    metrics["precision"] = get_precision_metric(y, y_pred)
    metrics["recall"] = get_recall_metric(y, y_pred)
    metrics["f1"] = get_f1_metric(y, y_pred)
    if DO_PRINT:
        print_evaluation_metrics(metrics)
    return metrics

def print_evaluation_metrics(metrics):
    """print the given summary of popular performance evaluation metrics that was caculated using get_evaluation_metrics"""
    print(f"accuracy = {metrics['accuracy']}")
    print(f"precision = {metrics['precision']}")
    print(f"recall = {metrics['recall']}")
    print(f"F1 = {metrics['f1']}")