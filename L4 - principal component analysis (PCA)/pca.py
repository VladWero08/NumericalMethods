import requests
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition._pca import PCA

def get_iris_dataset() -> dict:
    iris_db = requests.get("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data")
    iris_db_text = iris_db.text.split("\n")
    iris_db_data = {}

    for signal in iris_db_text:
        signal_values = signal.split(",")
        # transform string to float
        signal_values_float = [float(x) for x in signal_values[:-1]]
        # source of the signal
        signal_source = signal_values[-1]

        # append the new signal to the
        # dictionary of signals
        if not signal_source in iris_db_data:
            iris_db_data[signal_source] = []
        else:
            iris_db_data[signal_source].append(signal_values_float)

    # remove the empty key
    del iris_db_data[""]

    return iris_db_data

def pca_signal_lowering(dataset):
    pca = PCA(n_components=2)
    dataset_ = pca.fit_transform(dataset)

    # plot the new points obtained
    # after applying PCA
    plt.scatter(dataset_[:, 0], dataset_[:, 1])
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.show()

if __name__ == "__main__":
    iris = get_iris_dataset()

    for sources, signals in iris.items():
        singals = np.array(signals)
        pca_signal_lowering(signals)