import requests
import numpy as np
import random

import sklearn.svm
import sklearn.metrics as metrics


def get_iris_dataset() -> (list, list):
    """
    From the database of Iris flowers, return a tuple
    that contains two vectors:
    - 1st: the signals for each flower
    - 2nd: labels of the signals
    
    """
    iris_db = requests.get("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data")
    iris_db_text = iris_db.text.split("\n")
    iris_signals = [] 
    iris_labels = []


    for signal in iris_db_text:
        signal_values = signal.split(",")
        # transform string to float
        signal_values_float = [float(x) for x in signal_values[:-1]]
        # source of the signal
        signal_source = signal_values[-1]

        if signal_source == "":
            continue

        iris_signals.append(signal_values_float)
        iris_labels.append(signal_source)

    return (iris_signals, iris_labels)

def get_classification_sets(signals: list, labels: list) -> dict:
    """
    Given a list of Iris flowers signals and their label,
    randomly extract 100 flowers, and their label, for the training
    set and 50 flower for the testing set.

    Return a dictionary with two keys: "training" and "data, each one
    containing a tuple with the signals and the labels.
    """
    data = list(zip(signals, labels))
    random.shuffle(data)
    signals, labels = zip(*data)

    training_signals = signals[:100]
    testing_signals = signals[100:]

    training_labels = labels[:100]
    testing_labels = labels[100:]

    return {
        "training": {
            "signals": training_signals, 
            "labels": training_labels
        },
        "testing": {
            "signals": testing_signals, 
            "labels": testing_labels
        }
    }

def svm_linear(dataset: dict) -> None:
    """
    Classification of Iris dataset using the SVM
    with the linear kernel function.
    """
    clf = sklearn.svm.SVC(kernel='linear')
    clf.fit(dataset["training"]["signals"], dataset["training"]["labels"])
    predictions = clf.predict(dataset["testing"]["signals"])

    accuracy = metrics.accuracy_score(
        y_true=dataset["testing"]["labels"],
        y_pred=predictions
    )
    confusion_matrix = metrics.confusion_matrix(
        y_true=dataset["testing"]["labels"],
        y_pred=predictions
    )

    print(f"SVM with linear kernel:")
    print("------------------------")
    print(f"Accuracy: {accuracy}")
    print(f"Confusion matrix:")
    print(confusion_matrix)
    print()

def svm_rbf(dataset: dict) -> None:
    """
    Classification of Iris dataset using the SVM
    with the RBF kernel function.
    """
    clf = sklearn.svm.SVC(kernel='rbf')
    clf.fit(dataset["training"]["signals"], dataset["training"]["labels"])
    predictions = clf.predict(dataset["testing"]["signals"])

    accuracy = metrics.accuracy_score(
        y_true=dataset["testing"]["labels"],
        y_pred=predictions
    )
    confusion_matrix = metrics.confusion_matrix(
        y_true=dataset["testing"]["labels"],
        y_pred=predictions
    )

    print(f"SVM with RBF kernel:")
    print("------------------------")
    print(f"Accuracy: {accuracy}")
    print(f"Confusion matrix:")
    print(confusion_matrix)
    print()


iris_signals, iris_labels = get_iris_dataset()
iris_dataset = get_classification_sets(iris_signals, iris_labels)

svm_linear(iris_dataset)
svm_rbf(iris_dataset)
