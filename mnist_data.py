import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from time import time
from models import Logistic, DecisionTree
from comparison import plot_accs
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
mnist = tf.keras.datasets.mnist


NEGATIVE = 0
POSITIVE = 1
SAMPLES_NUM = 0
NEG_TITLE = "Labeled with 0"
POS_TITLE = "Labeled with 1"
MODELS_TITLES = ["Logistic Regression Accuracy", "Soft SVM Accuracy", "Decision Tree Accuracy", "KNN Accuracy"]
RUNTIME_PRINTS = ["Logistic Regression Mean Runtime:", "Soft SVM Mean Runtime:", "Decision Tree Mean Runtime:",
                  "KNN Mean Runtime:"]
NUM_OF_CLASSIFIERS = 4
LOG = 0
SVM = 1
TREE = 2
KNN = 3
M = [50, 100, 300, 500]


def load_data():
    """
    Loads images and labels from mnist and filters to images with binary labels
    :return: (x_train, y_train), (x_test, y_test) - 4 numpy arrays
            x_train is 12665 samples of 28x28 (shape (12665,28,28))
            x_test is 2115 samples of 28x28 (shape (2115,28,28))
            y_train with shape (12665,), y_test with shape (2115,)
    """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    train_images = np.logical_or((y_train == NEGATIVE), (y_train == POSITIVE))
    test_images = np.logical_or((y_test == NEGATIVE), (y_test == POSITIVE))

    x_train, y_train = x_train[train_images], y_train[train_images]
    x_test, y_test = x_test[test_images], y_test[test_images]

    return (x_train, y_train), (x_test, y_test)


def plot_images():  # Question 12
    """
    Loads images from mnist, and plots 3 images labeled with 1 and 3 images labeled with 0
    :return: None
    """
    (x_train, y_train) = load_data()[0]
    x_pos, x_neg = get_binary(3, x_train, y_train)

    for image in x_pos:
        plot_one_image(image, POS_TITLE)
    for image in x_neg:
        plot_one_image(image, NEG_TITLE)


def get_binary(amount, x, y):
    """
    Returns images with both 0 and 1 label, given amount of images from each label
    :param amount: int - number of images from 0 and 1
    :param x: Samples - numpy array of shape (m, n, n) (m images of shape nxn)
    :param y: Labels - numpy array of shape (m,)
    :return: x_pos, x_neg : 2 numpy arrays of shape (amount, n, n)
             Note: if amount > m, returns the arguments as is
    """
    x_pos = x[y == POSITIVE]
    x_neg = x[y == NEGATIVE]

    pos_amont = np.min([x_pos.shape[SAMPLES_NUM], amount])
    neg_amont = np.min([x_neg.shape[SAMPLES_NUM], amount])

    return x_pos[:pos_amont], x_neg[:neg_amont]


def plot_one_image(image, title):
    """
    Plots given image with a given title
    :param image: 2-d numpy array
    :param title: string of the title
    :return: None
    """
    plt.figure()
    plt.title(title)
    plt.imshow(image)
    plt.show()


def rearrange_data(X):
    """
    :param X: A tensor of shape (m, n, n)
    :return: X reshaped for (m, n^2)
    """
    return X.reshape(X.shape[SAMPLES_NUM], -1)


def eval_accuracy2():  # Question 14
    (x_train, y_train), (x_test, y_test) = load_data()
    x_test = rearrange_data(x_test)
    accuracies = np.zeros((NUM_OF_CLASSIFIERS, 1))
    runtime_eval = np.zeros((NUM_OF_CLASSIFIERS, 1))

    for m in M:
        culm_acc = np.zeros((NUM_OF_CLASSIFIERS, 1))
        runtime = np.zeros((NUM_OF_CLASSIFIERS, 1))

        for i in range(50):
            x, y = draw_samples(x_train, y_train, m)
            models = get_models2()

            for j in range(NUM_OF_CLASSIFIERS):
                log_tree = True if j == LOG or j == TREE else False
                acc, elapsed_time = fit_and_eval(models[j], x, y, x_test, y_test, log_tree)
                culm_acc[j] += acc
                runtime[j] += elapsed_time

        culm_acc /= 50
        runtime /= 50
        accuracies = np.concatenate((accuracies, culm_acc), axis=1)
        runtime_eval = np.concatenate((runtime_eval, runtime), axis=1)

    plot_accs(accuracies[:, 1:], MODELS_TITLES, NUM_OF_CLASSIFIERS, M)
    print_runtime(runtime_eval[:, 1:])


def draw_samples(x, y, m):
    """
    Draws m samples randomly from x and y, ensuring both labels o and 1 are included in the drawn set
    :param x: A tensor of shape (samples_num, n, n)
    :param y: a numpy array of shape (samples_num,)
    :param m: int between 0 to samples_num
    :return: x_draw, y_draw the drawn arrays, after x war rearranged to shape (samples_num, n^2)
    """
    indices = np.random.choice(x.shape[SAMPLES_NUM], m, replace=False)
    y_draw = y[indices]

    while POSITIVE not in y_draw or NEGATIVE not in y_draw:
        indices = np.random.choice(x.shape[SAMPLES_NUM], m, replace=False)
        y_draw = y[indices]

    x_draw = rearrange_data(x[indices])
    return x_draw, y_draw


def fit_and_eval(model, x_train, y_train, x_test, y_test, log_tree):
    """
    Fits a model, evaluating its accuracy and measure the elapsed time of the process
    :param model: A learning model, should implement fit and score as in sklearn documentation
    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :param log_tree: True if the model is a logistic regression of decision tree classifier
    :return:
    """
    if log_tree:
        start = time()
        model.fit(x_train.T, y_train)
        acc = model.score(x_test.T, y_test)['accuracy']
    else:
        start = time()
        model.fit(x_train, y_train)
        acc = model.score(x_test, y_test)

    end = time()

    return acc, end - start


def get_models2():
    """
    Creates 4 new classifiers of type Logistic, Soft-SVM, DecisionTree and k-nearest neighbors
    :return: array of Logistic, Soft-SVM, DecisionTree and k-nearest neighbors unfitted models
    """
    log = Logistic()
    svm = SVC(C=1, kernel="linear")
    tree = DecisionTree()
    knn = KNeighborsClassifier(n_neighbors=2, weights='distance', algorithm='brute')

    return [log, svm, tree, knn]


def print_runtime(runtime_arr):
    """
    Prints the runtimes in the given array
    :param runtime_arr: 2-d array
    :return: None
    """
    for i in range(NUM_OF_CLASSIFIERS):
        print(RUNTIME_PRINTS[i])
        for j in range(len(M)):
            print(M[j], "Samples:", runtime_arr[i][j])
        print()

