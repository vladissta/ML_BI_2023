import numpy as np
import pandas as pd


class KNNClassifier:
    """
    K-neariest-neighbor classifier using L1 loss
    """

    def __init__(self, k=1):
        self.k = k

    def fit(self, X, y):
        self.train_X = X
        self.train_y = y

    def predict(self, X, n_loops=0):
        """
        Uses the KNN model to predict clases for the data samples provided
        
        Arguments:
        X, np array (num_samples, num_features) - samples to run
           through the model
        num_loops, int - which implementation to use

        Returns:
        predictions, np array of ints (num_samples) - predicted class
           for each sample
        """

        if n_loops == 0:
            distances = self.compute_distances_no_loops(X)
        elif n_loops == 1:
            distances = self.compute_distances_one_loops(X)
        else:
            distances = self.compute_distances_two_loops(X)

        if len(np.unique(self.train_y)) == 2:
            return self.predict_labels_binary(distances)
        else:
            return self.predict_labels_multiclass(distances)

    def compute_distances_two_loops(self, X):
        """
        Computes L1 distance from every sample of X to every training sample
        Uses simplest implementation with 2 Python loops

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        """
        distances_array = np.zeros(shape=(len(X), len(self.train_X)))

        for n_train, x_train in enumerate(self.train_X):
            for n_test, x_test in enumerate(X):
                distances_array[n_test, n_train] = sum(abs(x_test - x_train))

        return distances_array

    def compute_distances_one_loop(self, X):
        """
        Computes L1 distance from every sample of X to every training sample
        Vectorizes some of the calculations, so only 1 loop is used

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        """

        distances_array = np.zeros(shape=(len(X), len(self.train_X)))

        for n_train, x_train in enumerate(self.train_X):
            distances_array[:, n_train] = abs((X - x_train)).sum(axis=1)

        return distances_array

    def compute_distances_no_loops(self, X):
        """
        Computes L1 distance from every sample of X to every training sample
        Fully vectorizes the calculations using numpy

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        """

        distances_array = np.sum(np.abs(X[:, None] - self.train_X[:]), axis=-1)

        return distances_array

    def predict_labels_binary(self, distances):
        """
        Returns model predictions for binary classification case
        
        Arguments:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        Returns:
        pred, np array of bool (num_test_samples) - binary predictions 
           for every test sample
        """

        # n_train = distances.shape[1]
        # n_test = distances.shape[0]
        # prediction = np.zeros(n_test)

        argmin_dists = distances.argsort(axis=1)
        min_k = argmin_dists[:, :self.k]
        neighbours = self.train_y[min_k]
        prediction = neighbours.astype('int64').mean(axis=1).round().astype('int64')

        return prediction

    def predict_labels_multiclass(self, distances):
        """
        Returns model predictions for multi-class classification case
        
        Arguments:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        Returns:
        pred, np array of int (num_test_samples) - predicted class index 
           for every test sample
        """

        argmin_dists = distances.argsort(axis=1)
        min_k = argmin_dists[:, :self.k]

        neighbours = self.train_y[min_k]

        neighbours_series = pd.DataFrame(neighbours)

        neighbours_freq = neighbours_series.apply(pd.Series.value_counts, axis=1).fillna(0)

        prediction = neighbours_freq.idxmax(axis=1)

        return prediction.to_numpy()
