from functools import partial

from numpy import linspace, full
from sklearn import *

from src.training.optimize import *

# https://scikit-learn.org/stable/modules/classes.html


def test_svc():
    optimize_parameters(
        partial(svm.SVC, kernel="linear"),
        lambda df: preprocessing.StandardScaler().fit_transform(df),
        C=linspace(0.001, 10, 100),
    ).display("SVM (C-support), linear")

    optimize_parameters(
        partial(svm.SVC, kernel="poly", degree=2),
        lambda df: preprocessing.StandardScaler().fit_transform(df),
        C=linspace(0.001, 10, 20),
        coef0=linspace(-100, 100, 20)
    ).display("SVM (C-support), polynomial (quadratic)")

    optimize_parameters(
        partial(svm.SVC, kernel="poly", degree=3),
        lambda df: preprocessing.StandardScaler().fit_transform(df),
        C=linspace(0.001, 10, 20),
        coef0=linspace(-100, 100, 20)
    ).display("SVM (C-support), polynomial (cubic)")

    optimize_parameters(
        partial(svm.SVC, kernel="poly", degree=4),
        lambda df: preprocessing.StandardScaler().fit_transform(df),
        C=linspace(0.001, 10, 20),
        coef0=linspace(-100, 100, 20)
    ).display("SVM (C-support), polynomial (quartic)")

    optimize_parameters(
        partial(svm.SVC, kernel="rbf"),
        lambda df: preprocessing.StandardScaler().fit_transform(df),
        C=linspace(0.001, 10, 100)
    ).display("SVM (C-support), RBF")

    optimize_parameters(
        partial(svm.SVC, kernel="sigmoid"),
        lambda df: preprocessing.StandardScaler().fit_transform(df),
        C=linspace(0.001, 10, 100)
    ).display("SVM (C-support), sigmoid")


def test_nu_svc():
    optimize_parameters(
        partial(svm.NuSVC, kernel="linear"),
        lambda df: preprocessing.StandardScaler().fit_transform(df),
        nu=linspace(0.001, 1, 100),
    ).display("SVM (Nu-support), linear")

    optimize_parameters(
        partial(svm.NuSVC, kernel="poly", degree=2),
        lambda df: preprocessing.StandardScaler().fit_transform(df),
        nu=linspace(0.001, 1, 20),
        coef0=linspace(-100, 100, 20)
    ).display("SVM (Nu-support), polynomial (quadratic)")

    optimize_parameters(
        partial(svm.NuSVC, kernel="poly", degree=3),
        lambda df: preprocessing.StandardScaler().fit_transform(df),
        nu=linspace(0.001, 1, 20),
        coef0=linspace(-100, 100, 20)
    ).display("SVM (Nu-support), polynomial (cubic)")

    optimize_parameters(
        partial(svm.NuSVC, kernel="poly", degree=4),
        lambda df: preprocessing.StandardScaler().fit_transform(df),
        nu=linspace(0.001, 1, 20),
        coef0=linspace(-100, 100, 20)
    ).display("SVM (Nu-support), polynomial (quartic)")

    optimize_parameters(
        partial(svm.NuSVC, kernel="rbf"),
        lambda df: preprocessing.StandardScaler().fit_transform(df),
        nu=linspace(0.001, 1, 100)
    ).display("SVM (Nu-support), RBF")

    optimize_parameters(
        partial(svm.NuSVC, kernel="sigmoid"),
        lambda df: preprocessing.StandardScaler().fit_transform(df),
        nu=linspace(0.001, 1, 100)
    ).display("SVM (Nu-support), sigmoid")


# Linear SVC does not converge.


def test_linear_model():
    optimize_parameters(
        linear_model.PassiveAggressiveClassifier,
        C=linspace(0.001, 10, 100)
    ).display("Linear Model (passive aggressive)")

    optimize_parameters(
        linear_model.RidgeClassifier,
        alpha=linspace(0.001, 10, 100)
    ).display("Linear Model (ridge)")

    # squared_error and squared_epsilon_insensitive do not converge in any reasonable number of iterations.
    optimize_parameters(
        partial(linear_model.SGDClassifier),
        loss=("hinge", "log_loss", "modified_huber", "squared_hinge",
              "perceptron", "huber", "epsilon_insensitive"),
        alpha=linspace(0.001, 10, 25)
    ).display("Linear Model (passive aggressive)")


def test_naive_bayes():
    optimize_parameters(
        naive_bayes.BernoulliNB,
        alpha=linspace(0.001, 10, 100)
    ).display("Naive Bayes (Bernoulli)")

    # Except for the Bernoulli model, Naive Bayes does not work with negative values.


def test_neighbors():
    optimize_parameters(
        neighbors.KNeighborsClassifier,
        n_neighbors=range(1, 100),
    ).display("k Nearest Neighbors")

    optimize_parameters(
        neighbors.RadiusNeighborsClassifier,
        radius=linspace(0.001, 10, 100),
        weights=("uniform", "distance")
    ).display("Radius Neighbors")


def test_neural_network():
    # There are a lot of parameters to vary here, so this will be expensive.
    # To reduce this somewhat, let all layers have the same amount of neurons.
    # TODO: try with varying amounts of neurons
    # lbfgs and sgd do not converge in any reasonable number of iterations.
    optimize_parameters(
        neural_network.MLPClassifier,
        hidden_layer_sizes=tuple(full(n, k) for n in range(1, 10) for k in range(1, 100, 10)),
        activation=("identity", "logistic", "tanh", "relu"),
        alpha=linspace(0.001, 10, 10)
    ).display("Neural Network (MLP)")


def test_tree():
    # TODO: vary other parameters
    optimize_parameters(
        tree.DecisionTreeClassifier,
        criterion=("gini", "entropy", "log_loss"),
        splitter=("best", "random")
    ).display("Decision Tree")

    # TODO: vary other parameters
    optimize_parameters(
        tree.DecisionTreeClassifier,
        criterion=("gini", "entropy", "log_loss"),
        splitter=("best", "random")
    ).display("Extra Tree")


# TODO: ensembles
