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
        partial(svm.SVC, kernel="poly", degree=2),
        "SVM (C-support), polynomial (quadratic)",
        lambda df: preprocessing.StandardScaler().fit_transform(df),
        C=linspace(0.065, 0.075, 50),
        coef0=linspace(30.5, 31, 50)
    )

    yield optimize_parameters(
        partial(svm.SVC, kernel="poly", degree=3),
        "SVM (C-support), polynomial (cubic)",
        lambda df: preprocessing.StandardScaler().fit_transform(df),
        C=linspace(0.00055, 0.00058, 50),
        coef0=linspace(50.4, 50.45, 50)
    )

    yield optimize_parameters(
        partial(svm.SVC, kernel="poly", degree=4),
        "SVM (C-support), polynomial (quartic)",
        lambda df: preprocessing.StandardScaler().fit_transform(df),
        C=linspace(0.00001, 0.000014, 25),
        coef0=linspace(42, 44, 25)
    )


def test_nu_svc() -> Iterator[OptimizeResult]:
    yield optimize_parameters(
        partial(svm.NuSVC, kernel="poly", degree=2),
        "SVM (Nu-support), polynomial (quadratic)",
        lambda df: preprocessing.StandardScaler().fit_transform(df),
        nu=linspace(0.00005, 0.0001, 25),
        coef0=linspace(4.7, 4.9, 25)
    )

    yield optimize_parameters(
        partial(svm.NuSVC, kernel="poly", degree=3),
        "SVM (Nu-support), polynomial (cubic)",
        lambda df: preprocessing.StandardScaler().fit_transform(df),
        nu=linspace(5.1e-6, 5.3e-6, 25),
        coef0=linspace(24.1, 24.3, 25)
    )

    yield optimize_parameters(
        partial(svm.NuSVC, kernel="poly", degree=4),
        "SVM (Nu-support), polynomial (quartic)",
        lambda df: preprocessing.StandardScaler().fit_transform(df),
        nu=linspace(5.4e-8, 5.42e-8, 25),
        coef0=linspace(30, 30.2, 25)
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


# TODO: ensembles
