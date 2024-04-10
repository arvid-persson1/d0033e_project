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
        C=linspace(0.5, 0.15, 100),
    )

    yield optimize_parameters(
        partial(svm.SVC, kernel="poly", degree=2),
        "SVM (C-support), polynomial (quadratic)",
        lambda df: preprocessing.StandardScaler().fit_transform(df),
        C=linspace(0.4, 0.6, 20),
        coef0=linspace(10, 20, 20)
    )

    yield optimize_parameters(
        partial(svm.SVC, kernel="poly", degree=3),
        "SVM (C-support), polynomial (cubic)",
        lambda df: preprocessing.StandardScaler().fit_transform(df),
        C=linspace(0.00001, 0.01, 20),
        coef0=linspace(30, 40, 20)
    )

    yield optimize_parameters(
        partial(svm.SVC, kernel="poly", degree=4),
        "SVM (C-support), polynomial (quartic)",
        lambda df: preprocessing.StandardScaler().fit_transform(df),
        C=linspace(0.00001, 0.01, 20),
        coef0=linspace(30, 40, 20)
    )

    yield optimize_parameters(
        partial(svm.SVC, kernel="rbf"),
        "SVM (C-support), RBF",
        lambda df: preprocessing.StandardScaler().fit_transform(df),
        C=linspace(2, 3, 100)
    )

    yield optimize_parameters(
        partial(svm.SVC, kernel="sigmoid"),
        "SVM (C-support), sigmoid",
        lambda df: preprocessing.StandardScaler().fit_transform(df),
        C=linspace(5, 6, 100)
    )


def test_nu_svc() -> Iterator[OptimizeResult]:
    yield optimize_parameters(
        partial(svm.NuSVC, kernel="linear"),
        "SVM (Nu-support), linear",
        lambda df: preprocessing.StandardScaler().fit_transform(df),
        nu=linspace(0.25, 0.4, 100),
    )

    yield optimize_parameters(
        partial(svm.NuSVC, kernel="poly", degree=2),
        "SVM (Nu-support), polynomial (quadratic)",
        lambda df: preprocessing.StandardScaler().fit_transform(df),
        nu=linspace(0.00001, 0.01, 20),
        coef0=linspace(4.5, 6, 20)
    )

    yield optimize_parameters(
        partial(svm.NuSVC, kernel="poly", degree=3),
        "SVM (Nu-support), polynomial (cubic)",
        lambda df: preprocessing.StandardScaler().fit_transform(df),
        nu=linspace(0.00001, 0.01, 20),
        coef0=linspace(20, 30, 20)
    )

    yield optimize_parameters(
        partial(svm.NuSVC, kernel="poly", degree=4),
        "SVM (Nu-support), polynomial (quartic)",
        lambda df: preprocessing.StandardScaler().fit_transform(df),
        nu=linspace(0.00001, 0.01, 20),
        coef0=linspace(30, 40, 20)
    )

    yield optimize_parameters(
        partial(svm.NuSVC, kernel="rbf"),
        "SVM (Nu-support), RBF",
        lambda df: preprocessing.StandardScaler().fit_transform(df),
        nu=linspace(0.01, 0.03, 100)
    )

    yield optimize_parameters(
        partial(svm.NuSVC, kernel="sigmoid"),
        "SVM (Nu-support), sigmoid",
        lambda df: preprocessing.StandardScaler().fit_transform(df),
        nu=linspace(0.25, 0.4, 100)
    )


# Linear SVC does not converge.


def test_linear_model() -> Iterator[OptimizeResult]:
    yield optimize_parameters(
        linear_model.PassiveAggressiveClassifier,
        "Linear Model (passive aggressive)",
        C=linspace(5, 6.5, 100)
    )

    yield optimize_parameters(
        linear_model.RidgeClassifier,
        "Linear Model (ridge)",
        alpha=linspace(0.00001, 0.01, 100)
    )

    # squared_error and squared_epsilon_insensitive do not converge in any reasonable number of iterations.
    yield optimize_parameters(
        partial(linear_model.SGDClassifier),
        "Linear Model (stochasitc gradient descent)",
        loss=("hinge", "log_loss", "modified_huber", "squared_hinge",
              "perceptron", "huber", "epsilon_insensitive"),
        alpha=linspace(0.00001, 0.01, 25)
    )


def test_naive_bayes() -> Iterator[OptimizeResult]:
    yield optimize_parameters(
        naive_bayes.BernoulliNB,
        "Naive Bayes (Bernoulli)",
        alpha=linspace(0.00001, 0.01, 100)
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
        radius=linspace(1.5, 3, 100),
        weights=("uniform", "distance")
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
        tree.DecisionTreeClassifier,
        "Decision Tree",
        criterion=("gini", "entropy", "log_loss"),
        splitter=("best", "random"),
        max_depth=range(1, 20),
        min_samples_split=linspace(0.01, 0.2, 20)
    )

    # TODO: vary other parameters
    yield optimize_parameters(
        tree.ExtraTreeClassifier,
        "Extra Tree",
        criterion=("gini", "entropy", "log_loss"),
        splitter=("best", "random"),
        max_depth=range(1, 20),
        min_samples_split=linspace(0.01, 0.2, 20)
    )


# TODO: ensembles
