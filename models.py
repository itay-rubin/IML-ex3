from abc import ABC, abstractmethod
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

ROWS = 0
COLS = 1
FIRST = 0
POSITIVE = 1
NEGATIVE = -1
NO_LABEL = 0
MU_POS = 0
MU_NEG = 1
SIGMA = 2
PR = 3
SAMPLES_NUM = 1


class Model(ABC):
    """
    An abstract class for a Classification model
    """

    @abstractmethod
    def fit(self, X, y):
        """
        Fits the model with a given training set.
        Stores the model in self.model
        :param X: The training samples - numpy array of shape (d x m)
        :param y: The labels of X - numpy array of shape (m,)
        :return: None
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Given an unlabeled test set X, predicts the label of each sample
        :param X: numpy array of shape (d x m)
        :return: y_hat - numpy array of shape (m,)
        """
        pass

    def score(self, X, y):
        """
        Calculates parameters of quality of the model for set of samples X
        :param X: Unlabeled test set of m' samples - numpy array of shape (d x m')
        :param y: true labels of the set - numpy array of shape (m',)
        :return: a dictionary of (num_samples, "error", accuracy, FPR, TPR, precision, recall)
        """
        y_hat = self.predict(X)
        tp, tn, fp, fn = self.__get_pred_stats(y, y_hat)

        pos_amount = y[y > 0].size
        neg_amount = y.size - pos_amount

        err_rate = 0 if y.size == 0 else (fp + fn) / y.size
        acc = 0 if y.size == 0 else (tp + tn) / y.size
        fpr = 0 if fp == 0 else fp / neg_amount
        tpr = 0 if tp == 0 else tp / pos_amount
        precision = 0 if tp == 0 else tp / (tp + fp)
        recall = 0 if tp == 0 else tp / pos_amount

        return {"num_samples": y.size, "error": err_rate, "accuracy": acc, "FPR": fpr, "TPR": tpr,
                "precision": precision, "recall": recall}

    @staticmethod
    def __get_pred_stats(y, y_hat):
        """
        Calculates stats for measuring the model quality: True positives, True negatives,
                                                          False positives, False negatives
        :param y: true labels, array of shape (m,)
        :param y_hat: estimated labels, array of shape (m,)
        :return: 4 numpy arrays in the shape of y: TP, TN, FP, FN
        """
        if y.size == 0:
            return 0, 0, 0, 0

        correct_pos = y_hat[y > 0]
        correct_neg = y_hat[y <= 0]
        est_pos = y[y_hat > 0]
        est_neg = y[y_hat <= 0]

        tp = np.count_nonzero(correct_pos > 0)
        tn = np.count_nonzero(correct_neg <= 0)
        fp = np.count_nonzero(est_pos <= 0)
        fn = np.count_nonzero(est_neg > 0)

        return tp, tn, fp, fn


class Perceptron(Model):
    """
    A classifier implementing the Perceptron learning algorithm
    """

    def __init__(self):
        """
        Initializing the model with an empty vector for the weights
        """
        self.model = np.array([])

    def fit(self, X, y):
        """
        Fits the model with a given training set
        Stores the weights vector in self.model
        :param X: The training samples - numpy array of shape (d x m)
        :param y: The labels of X - numpy array of shape (m,)
        :return: None
        """
        train_set = np.concatenate((np.ones_like(y).reshape(1, -1), X))
        self.model = np.zeros((train_set.shape[ROWS], 1))
        y_hat = np.zeros_like(y)
        constrains = (y * y_hat) <= 0

        while constrains.any():
            fix_idx = np.where(constrains)[FIRST][FIRST]  # np.where returns tuple size 1
            self.model += (y[fix_idx] * train_set.T[fix_idx]).reshape(self.model.shape)
            y_hat = self.model.T @ train_set
            constrains = np.squeeze((y * y_hat) <= 0)

    def predict(self, X):
        """
        Given an unlabeled test set X, predicts the label of each sample
        :param X: numpy array of shape (d x m)
        :return: y_hat - numpy array of shape (m,)
        """
        test_set = np.concatenate((np.ones(X.shape[COLS]).reshape(1, -1), X))
        y_hat = np.sign(self.model.T @ test_set)
        y_hat = np.where(y_hat == NO_LABEL, NEGATIVE, y_hat)
        return y_hat.squeeze()


class LDA(Model):
    """
    A class for LDA classifier
    """

    def __init__(self):
        """
        Initializes the model with an empty array
        """
        self.model = []

    def fit(self, X, y):
        """
        Fits the model with a given training set.
        Stores the array [positive mean, negative mean, Sigma^-1, Pr(y=1)] in self.model
        :param X: The training samples - numpy array of shape (d x m)
        :param y: The labels of X - numpy array of shape (m,)
        :return: None
        """
        pos_pr = self.__get_positive_p(y)
        mu_pos, mu_neg, inv_sigma = self.__get_statistics(X, y)

        self.model = [mu_pos, mu_neg, inv_sigma, pos_pr]

    def predict(self, X):
        """
        Given an unlabeled test set X, predicts the label of each sample
        :param X: numpy array of shape (d x m)
        :return: y_hat - numpy array of shape (m,)
        """
        pos_delta, neg_delta = self.__get_delta(X)

        compare = pos_delta > neg_delta
        y_hat = np.where(compare, POSITIVE, NEGATIVE)

        return y_hat.squeeze()

    def __get_delta(self, X):
        """
        Calculates the discriminant functions for given samples
        :param X: numpy array of shape (d x m)
        :return: positive delta, negative delta - numpy arrays of shape (m x 1)
        """
        mu_pos, mu_neg, inv_sigma, pos_pr = self.model

        pos_delta = X.T @ inv_sigma @ mu_pos - 0.5 * mu_pos.T @ inv_sigma @ mu_pos + np.log(pos_pr)
        neg_delta = X.T @ inv_sigma @ mu_neg - 0.5 * mu_neg.T @ inv_sigma @ mu_neg + np.log(1 - pos_pr)

        return pos_delta, neg_delta

    @staticmethod
    def __get_statistics(X, y):
        """
        Calculates the mean vectors for samples with positive labels and with negative labels
        :param X: The training samples - numpy array of shape (d x m)
        :param y: The labels of X - numpy array of shape (m,)
        :return: mu_pos, mu_neg, inv_sigma - means of the positive and negative labeled samples respectively,
                                             inverse covariance matrix

        """
        pos_samples = (X.T[y == POSITIVE]).T
        neg_samples = (X.T[y == NEGATIVE]).T

        mu_pos = pos_samples.mean(axis=COLS).reshape(-1, 1)
        mu_neg = neg_samples.mean(axis=COLS).reshape(-1, 1)

        inv_sigma = LDA.__get_inv_sigma(pos_samples, neg_samples, mu_pos, mu_neg, X.shape[SAMPLES_NUM])

        return mu_pos, mu_neg, inv_sigma

    @staticmethod
    def __get_positive_p(y):
        """
        Calculates the probability for a positive label
        :param y: array of +-1 labels in shape (num_samples,)
        :return: Pr(y=1)
        """
        pos_num = y[y == POSITIVE].size
        return pos_num / y.size

    @staticmethod
    def __get_inv_sigma(pos_x, neg_x, mu_pos, mu_neg, m):
        """
        Calculates the inverse of estimated covariance matrix using conditional distribution
        :param pos_x: Samples with a positive label - numpy array of shape (d, pos_samples)
        :param neg_x: Samples with a negative label - numpy array of shape (d, neg_samples)
        :param mu_pos: mean of the positive labeled samples - numpy array of shape (d,)
        :param mu_neg: mean of the negative labeled samples - numpy array of shape (d,)
        :param m: total number of samples (pos_samples + neg_samples)
        :return: inverse of the estimated covariance matrix - numpy array of shape (d, d)
        """
        pos_cov = (pos_x - mu_pos.reshape(-1, 1)) @ (pos_x - mu_pos.reshape(-1, 1)).T
        neg_cov = (neg_x - mu_neg.reshape(-1, 1)) @ (neg_x - mu_neg.reshape(-1, 1)).T
        est_cov = (pos_cov + neg_cov) / (m - 2)
        return np.linalg.inv(est_cov)


class SVM(Model):
    """
    A class for a Hard-SVM Classifier
    """

    def __init__(self):
        """
        Initializes the object with an object of sklearn.svm.svc and an empty weights vector
        """
        self.__svm = SVC(C=1e10, kernel="linear")
        self.model = np.array([])

    def fit(self, X, y):
        """
        Fits the model with a given training set.
        :param X: The training samples - numpy array of shape (d x m)
        :param y: The labels of X - numpy array of shape (m,)
        :return: None
        """

        self.__svm.fit(X.T, y)
        self.model = np.concatenate((self.__svm.intercept_, self.__svm.coef_.squeeze()))

    def predict(self, X):
        """
        Given an unlabeled test set X, predicts the label of each sample
        :param X: numpy array of shape (d x m)
        :return: y_hat - numpy array of shape (m,)
        """
        return self.__svm.predict(X.T)


class Logistic(Model):
    """
    A class for a Logistic Regression Classifier
    """

    def __init__(self):
        """
        Initializes the object with an object of sklearn.linear_model.LogisticRegression
        """
        self.__logistic = LogisticRegression(solver="liblinear")
        self.model = np.array([])

    def fit(self, X, y):
        """
        Fits the model with a given training set.
        :param X: The training samples - numpy array of shape (d x m)
        :param y: The labels of X - numpy array of shape (m,)
        :return: None
        """
        self.__logistic.fit(X.T, y)
        self.model = np.concatenate((self.__logistic.intercept_, self.__logistic.coef_.squeeze()))

    def predict(self, X):
        """
        Given an unlabeled test set X, predicts the label of each sample
        :param X: numpy array of shape (d x m)
        :return: y_hat - numpy array of shape (m,)
        """
        return self.__logistic.predict(X.T)


class DecisionTree(Model):
    """
    A class for a Decision Tree Classifier
    """

    def __init__(self):
        """
        Initializes the object with an object of sklearn.linear_model.LogisticRegression
        """
        self.__tree = DecisionTreeClassifier(max_depth=5)

    def fit(self, X, y):
        """
        Fits the model with a given training set.
        :param X: The training samples - numpy array of shape (d x m)
        :param y: The labels of X - numpy array of shape (m,)
        :return: None
        """
        self.__tree.fit(X.T, y)

    def predict(self, X):
        """
        Given an unlabeled test set X, predicts the label of each sample
        :param X: numpy array of shape (d x m)
        :return: y_hat - numpy array of shape (m,)
        """
        return self.__tree.predict(X.T) #


