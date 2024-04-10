from functools import partial
from json import JSONEncoder, dumps
from typing import Iterator

from numpy import linspace, full
from sklearn import *

from src.training.optimize import *


class OptimizeEncoder(JSONEncoder):
    def default(self, o):
        if isinstance(o, OptimizeResult):
            return vars(o)
        else:
            return super().default(o)


def dump(*results):
    print(dumps(results, cls=OptimizeEncoder, indent=4))


# https://scikit-learn.org/stable/modules/classes.html


def test_svc() -> Iterator[OptimizeResult]:
    yield optimize_parameters(
        partial(svm.SVC, kernel="linear"),
        "SVM (C-support), linear",
        preprocessor=lambda df: preprocessing.StandardScaler().fit_transform(df),
        C=linspace(0.0001, 0.05, 100),
    )

    yield optimize_parameters(
        partial(svm.SVC, kernel="poly", degree=2),
        "SVM (C-support), polynomial (quadratic)",
        lambda df: preprocessing.StandardScaler().fit_transform(df),
        C=linspace(0.0001, 0.4, 50),
        coef0=linspace(5, 15, 10)
    )

    yield optimize_parameters(
        partial(svm.SVC, kernel="poly", degree=3),
        "SVM (C-support), polynomial (cubic)",
        lambda df: preprocessing.StandardScaler().fit_transform(df),
        C=linspace(0.00001, 0.001, 50),
        coef0=linspace(30, 40, 10)
    )

    yield optimize_parameters(
        partial(svm.SVC, kernel="poly", degree=4),
        "SVM (C-support), polynomial (quartic)",
        lambda df: preprocessing.StandardScaler().fit_transform(df),
        C=linspace(0.00001, 0.0001, 20),
        coef0=linspace(30, 40, 20)
    )

    yield optimize_parameters(
        partial(svm.SVC, kernel="rbf"),
        "SVM (C-support), RBF",
        lambda df: preprocessing.StandardScaler().fit_transform(df),
        C=linspace(2.3, 2.6, 50)
    )

    yield optimize_parameters(
        partial(svm.SVC, kernel="sigmoid"),
        "SVM (C-support), sigmoid",
        lambda df: preprocessing.StandardScaler().fit_transform(df),
        C=linspace(5.6, 5.75, 50)
    )


def test_nu_svc() -> Iterator[OptimizeResult]:
    yield optimize_parameters(
        partial(svm.NuSVC, kernel="linear"),
        "SVM (Nu-support), linear",
        lambda df: preprocessing.StandardScaler().fit_transform(df),
        nu=linspace(0.3, 0.35, 50),
    )

    yield optimize_parameters(
        partial(svm.NuSVC, kernel="poly", degree=2),
        "SVM (Nu-support), polynomial (quadratic)",
        lambda df: preprocessing.StandardScaler().fit_transform(df),
        nu=linspace(0.00001, 0.0001, 20),
        coef0=linspace(4.75, 5.25, 10)
    )

    yield optimize_parameters(
        partial(svm.NuSVC, kernel="poly", degree=3),
        "SVM (Nu-support), polynomial (cubic)",
        lambda df: preprocessing.StandardScaler().fit_transform(df),
        nu=linspace(0.0000001, 0.00001, 50),
        coef0=linspace(20, 25, 10)
    )

    yield optimize_parameters(
        partial(svm.NuSVC, kernel="poly", degree=4),
        "SVM (Nu-support), polynomial (quartic)",
        lambda df: preprocessing.StandardScaler().fit_transform(df),
        nu=linspace(0.0000001, 0.00001, 50),
        coef0=linspace(27.5, 32.5, 20)
    )

    yield optimize_parameters(
        partial(svm.NuSVC, kernel="rbf"),
        "SVM (Nu-support), RBF",
        lambda df: preprocessing.StandardScaler().fit_transform(df),
        nu=linspace(0.01, 0.015, 50)
    )

    yield optimize_parameters(
        partial(svm.NuSVC, kernel="sigmoid"),
        "SVM (Nu-support), sigmoid",
        lambda df: preprocessing.StandardScaler().fit_transform(df),
        nu=linspace(0.3, 0.35, 50)
    )


# Linear SVC does not converge.


def test_linear_model() -> Iterator[OptimizeResult]:
    yield optimize_parameters(
        linear_model.PassiveAggressiveClassifier,
        "Linear Model (passive aggressive)",
        C=linspace(5.5, 6, 50)
    )

    yield optimize_parameters(
        linear_model.RidgeClassifier,
        "Linear Model (ridge)",
        alpha=linspace(0.001, 0.01, 50)
    )

    # squared_error and squared_epsilon_insensitive do not converge in any reasonable number of iterations.
    yield optimize_parameters(
        partial(linear_model.SGDClassifier, loss="squared hinge"),
        "Linear Model (stochasitc gradient descent)",
        alpha=linspace(0.001, 0.002, 50)
    )


def test_naive_bayes() -> Iterator[OptimizeResult]:
    yield optimize_parameters(
        naive_bayes.BernoulliNB,
        "Naive Bayes (Bernoulli)",
        alpha=linspace(0.0000001, 0.00001, 100)
    )

    # Except for the Bernoulli model, Naive Bayes does not work with negative values.


def test_neighbors() -> Iterator[OptimizeResult]:
    # Included only for completeness.
    yield optimize_parameters(
        partial(neighbors.KNeighborsClassifier, weights="uniform"),
        "k Nearest Neighbors",
        n_neighbors=(1,)
    )

    yield optimize_parameters(
        partial(neighbors.RadiusNeighborsClassifier, weights="distance"),
        "Radius Neighbors",
        radius=linspace(2, 2.5, 100),
    )


def test_neural_network() -> Iterator[OptimizeResult]:
    # There are a lot of parameters to vary here, so this will be expensive.
    # To reduce this somewhat, let all layers have the same amount of neurons.
    # TODO: try with varying amounts of neurons
    # lbfgs and sgd do not converge in any reasonable number of iterations.
    yield optimize_parameters(
        neural_network.MLPClassifier,
        "Neural Network (MLP)",
        hidden_layer_sizes=tuple(full(n, k) for n in range(1, 10) for k in range(1, 100, 10)),
        activation=("identity", "logistic", "tanh", "relu"),
        alpha=linspace(0.001, 10, 10)
    )


def test_tree() -> Iterator[OptimizeResult]:
    # TODO: vary other parameters
    yield optimize_parameters(
        partial(tree.DecisionTreeClassifier, criterion="log_loss", splitter="best", max_features=240),
        "Decision Tree",
        max_depth=range(10, 20),
        min_samples_split=range(1, 20),
        min_samples_leaf=range(5)
    )

    # TODO: vary other parameters
    yield optimize_parameters(
        partial(tree.ExtraTreeClassifier, criterion="gini", splitter="best", max_features=240),
        "Extra Tree",
        max_depth=range(10, 20),
        min_samples_split=range(1, 20),
        min_samples_leaf=range(5)
    )


# TODO: ensembles
