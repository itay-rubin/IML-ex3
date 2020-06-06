from models import *
import matplotlib.pyplot as plt

M = [5, 10, 15, 25, 70]
K = 10000
BIAS = 0
X = 1
Y = 2
HP_LABELS = ["True HP", "Perceptron HP", "SVM HP"]
MODELS_TITLES = ["Perceptron Accuracy", "SVM Accuracy", "LDA Accuracy"]


def draw_points(m):
    """
    Generating samples of 2-dimensional Gaussian distribution and their labels according to the function:
    y = sign(<(0.3, -0.5), X> + 0.1)
    :param m: int, number of samples to generate
    :return: X - the samples matrix, numpy array of shape (2, m)
             y - the labels of the samples from {+-1}, numpy array of shape (m,)
    """
    X = gen_samples(m)
    y = gen_labels(X)
    return X, y


def gen_samples(m):
    """
    Generates m samples of 2-dimensional Gaussian vector with mean 0 and var I
    :param m: int, number of samples
    :return: X - numpy array of shape (2, m)
    """
    mean = np.zeros(2)
    cov = np.eye(2)
    return np.random.multivariate_normal(mean, cov, m).reshape(2, m)


def gen_labels(X):
    """
    Generates labels for given samples according to y = sign(<(0.3, -0.5), X> + 0.1)
    :param X: numpy array of shape (2,m)
    :return: y, numpy array of shape (m,)
    """
    w = np.array([0.3, -0.5])
    return np.sign(w @ X + 0.1)


def gen_and_plot():  # Question 9
    """
    Generates points from multivariate normal distribution, fits Perceptron and SVM models and plots the points and
    the decision hyperplane of each model, and the hp of the "true" decision of creating the labels
    :return: None
    """
    x_axis = np.linspace(-5, 5, 11)
    true_hp = 0.6 * x_axis + 0.2

    for m in M:
        X, y = draw_points(m)
        while not np.sum(y == POSITIVE) or not np.sum(y == NEGATIVE):
            X, y = draw_points(m)

        perc, svm = get_models(lda=False)
        fit_models(X, y, perc, svm)

        perc_hp, svm_hp = get_hyperplanes(perc, svm, x_axis)
        pos_pts = X.T[y > 0].T
        neg_pts = X.T[y < 0].T
        plot_hps(x_axis, pos_pts, neg_pts, [true_hp, perc_hp, svm_hp], m)


def get_models(lda=True):
    """
    Creates 2 or 3 new classifiers of type Perceptron, Hard-SVM and LDA
    :param lda: it True generates an LDA model as well
    :return: perceptron, svm, (lda) - 2(3) objects of class Perceptron, SVM, (LDA) respectively
    """
    perc = Perceptron()
    svm = SVM()

    if lda:
        lda = LDA()
        return perc, svm, lda
    else:
        return perc, svm


def fit_models(X, y, perceptron, svm, lda=None):
    """
    Fits 2 models of class Perceptron, SVM
    :param X: The training samples - numpy array of shape (d x m)
    :param y: The labels of X - numpy array of shape (m,)
    :param perceptron: Perceptron model
    :param svm: SVM model
    :param lda: LDA model (optional)
    :return: None
    """
    perceptron.fit(X, y)
    svm.fit(X, y)
    if lda is not None:
        lda.fit(X, y)


def get_hyperplanes(perceptron, svm, x_axis):
    """
    Calculates the decision hyperplane equation for d = 2 for Perceptron and SVM classifiers
    :param perceptron: A fitted model of class Perceptron
    :param svm: A fitted model of class SVM
    :param x_axis: np array for x axis of the line equation
    :return: Perceptron hyperplane, SVM hyperplane respectively as a numpy array
    """
    w_perc, w_svm = perceptron.model, svm.model
    perc_hp = (-w_perc[X] * x_axis - w_perc[BIAS]) / w_perc[Y]
    svm_hp = (-w_svm[X] * x_axis - w_svm[BIAS]) / w_svm[Y]
    return perc_hp, svm_hp


def plot_hps(x_axis, pos, neg, hps, m):
    """
    Plots a graph with the hyperplanes (lines) in hps and points in pos and neg
    :param x_axis: np array for x axis of the line equation
    :param pos: Points with positive label
    :param neg: Points with negative label
    :param hps: A Python array: [true_hp, perc_hp, svm_hp], each hyperplane is a 1-d numpy array
    :param m: int, number of samples
    :return: None
    """
    plt.figure()
    plt.scatter(pos[0], pos[1], s=10, c='b', marker='.')
    plt.scatter(neg[0], neg[1], s=10, c='#f97306', marker='.')

    for i in range(3):
        plt.plot(x_axis, hps[i], label=HP_LABELS[i])

    plt.title('Generated Points and the Decision Hyperplanes, m = ' + str(m))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='upper left')
    plt.grid()
    plt.show()


def eval_accuracy():  # Question 10
    """
    Creates and trains Perceptron, SVM and LDA models iteratively with points from multivariate normal distribution and
    plots the mean accuracy of each model for labeling new points for each number of samples
    :return: None
    """
    perc_mean_acc = []
    svm_mean_acc = []
    lda_mean_acc = []

    for m in M:
        perc_acc = svm_acc = lda_acc = 0
        for i in range(500):
            X, y = draw_points(m)
            while not np.sum(y == POSITIVE) or not np.sum(y == NEGATIVE):
                X, y = draw_points(m)

            X_test, y_test = draw_points(K)

            perc, svm, lda = get_models()
            fit_models(X, y, perc, svm, lda)

            perc_temp, svm_temp, lda_temp = get_accs(X_test, y_test, perc, svm, lda)
            perc_acc += perc_temp
            svm_acc += svm_temp
            lda_acc += lda_temp

        perc_mean_acc.append(perc_acc / 500)
        svm_mean_acc.append(svm_acc / 500)
        lda_mean_acc.append(lda_acc / 500)

    plot_accs(np.array([perc_mean_acc, svm_mean_acc, lda_mean_acc]), MODELS_TITLES, 3, M)


def get_accs(X, y, perceptron, svm, lda):
    """
    Evaluates the models on a new test set
    :param X: The test samples - numpy array of shape (d x k)
    :param y: The labels of X - numpy array of shape (k,)
    :param perceptron: Perceptron model
    :param svm: SVM model
    :param lda: LDA model
    :return: perceptron accuracy, svm accuracy, lda accuracy respectively
    """
    perc_acc = perceptron.score(X, y)['accuracy']
    svm_acc = svm.score(X, y)['accuracy']
    lda_acc = lda.score(X, y)['accuracy']
    return perc_acc, svm_acc, lda_acc


def plot_accs(accs, labels, length, x_axis):
    """
    Plots the arrays
    :param accs: Array of the accuracies to plot
    :param labels: Labels for the plot, same length as accs
    :param length: int, len of the arrays
    :param x_axis: array for the x axis of the graph
    :return: None
    """
    plt.figure()
    for i in range(length):
        plt.plot(x_axis, accs[i], label=labels[i])

    plt.title('Accuracies of Classifiers')
    plt.xlabel('m')
    plt.ylabel('Mean Accuracy')
    plt.xlim(x_axis[0] - 1, x_axis[-1] + 1)
    plt.ylim(accs.min() - 0.01, accs.max() + 0.005)
    plt.legend(loc='lower right')
    plt.grid()

    plt.show()

