from functools import partial
from typing import Iterator

from numpy import linspace, full
from sklearn import *

from optimize import optimize_parameters, OptimizeResult

# https://scikit-learn.org/stable/modules/classes.html


def test_svm() -> Iterator[OptimizeResult]:
    yield optimize_parameters(
        partial(svm.NuSVC, kernel="linear"),
        "SVM, linear",
        lambda df: preprocessing.StandardScaler().fit_transform(df),
        nu=linspace(0.001, 1, 100),
    )

    yield optimize_parameters(
        partial(svm.NuSVC, kernel="poly", degree=2),
        "SVM, polynomial (quadratic)",
        lambda df: preprocessing.StandardScaler().fit_transform(df),
        nu=linspace(0.001, 1, 20),
        coef0=linspace(-100, 100, 20)
    )

    yield optimize_parameters(
        partial(svm.NuSVC, kernel="poly", degree=3),
        "SVM, polynomial (cubic)",
        lambda df: preprocessing.StandardScaler().fit_transform(df),
        nu=linspace(0.001, 1, 20),
        coef0=linspace(-100, 100, 20)
    )

    yield optimize_parameters(
        partial(svm.NuSVC, kernel="poly", degree=4),
        "SVM, polynomial (quartic)",
        lambda df: preprocessing.StandardScaler().fit_transform(df),
        nu=linspace(0.001, 1, 20),
        coef0=linspace(-100, 100, 20)
    )

    yield optimize_parameters(
        partial(svm.NuSVC, kernel="rbf"),
        "SVM, RBF",
        lambda df: preprocessing.StandardScaler().fit_transform(df),
        nu=linspace(0.001, 1, 100)
    )

    yield optimize_parameters(
        partial(svm.NuSVC, kernel="sigmoid"),
        "SVM, sigmoid",
        lambda df: preprocessing.StandardScaler().fit_transform(df),
        nu=linspace(0.001, 1, 100)
    )


def test_linear_model() -> Iterator[OptimizeResult]:
    yield optimize_parameters(
        linear_model.PassiveAggressiveClassifier,
        "Linear Model (passive aggressive)",
        C=linspace(0.001, 10, 100)
    )

    yield optimize_parameters(
        linear_model.RidgeClassifier,
        "Linear Model (ridge)",
        alpha=linspace(0.001, 10, 100)
    )

    # squared_error and squared_epsilon_insensitive do not converge in any reasonable number of iterations.
    yield optimize_parameters(
        partial(linear_model.SGDClassifier),
        "Linear Model (passive aggressive)",
        loss=("hinge", "log_loss", "modified_huber", "squared_hinge",
              "perceptron", "huber", "epsilon_insensitive"),
        alpha=linspace(0.001, 10, 25)
    )


def test_naive_bayes() -> Iterator[OptimizeResult]:
    yield optimize_parameters(
        naive_bayes.BernoulliNB,
        "Naive Bayes (Bernoulli)",
        alpha=linspace(0.001, 10, 100)
    )

    # Except for the Bernoulli model, Naive Bayes does not work with negative values.


def test_neighbors() -> Iterator[OptimizeResult]:
    yield optimize_parameters(
        neighbors.KNeighborsClassifier,
        "k Nearest Neighbors",
        n_neighbors=range(1, 100),
        weights=("uniform", "distance")
    )

    yield optimize_parameters(
        neighbors.RadiusNeighborsClassifier,
        "Radius Neighbors",
        radius=linspace(0.001, 10, 100),
        weights=("uniform", "distance")
    )


def test_tree() -> Iterator[OptimizeResult]:
    yield optimize_parameters(
        partial(tree.DecisionTreeClassifier, criterion="log_loss", splitter="best", max_features=240),
        "Decision Tree",
        max_depth=range(10, 20),
        min_samples_split=range(1, 20),
        min_samples_leaf=range(5)
    )

    yield optimize_parameters(
        partial(tree.ExtraTreeClassifier, criterion="gini", splitter="best", max_features=240),
        "Extra Tree",
        max_depth=range(10, 20),
        min_samples_split=range(1, 20),
        min_samples_leaf=range(5)
    )


def test_neural_network() -> Iterator[OptimizeResult]:
    # lbfgs. sgd, identity and relu do not converge in any reasonable number of iterations.
    yield optimize_parameters(
        # Even with this amount of iterations, sometimes warnings are raised.
        # This is difficult to avoid without compromising accuracy.
        partial(neural_network.MLPClassifier, max_iter=2500),
        "Neural Network (MLP)",
        # FIXME: type error
        hidden_layer_sizes=tuple(full(n, k) for n in range(1, 5) for k in range(1, 300, 20)),
        activation=("logistic", "tanh"),
        alpha=linspace(0.001, 10, 10)
    )


# TODO: ensembles
