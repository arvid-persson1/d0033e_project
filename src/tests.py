from functools import partial
from typing import Iterator

from numpy import linspace, full
from sklearn import *

from optimize import optimize_parameters, OptimizeResult

SEED = 1


# https://scikit-learn.org/stable/modules/classes.html


def test_svm() -> Iterator[OptimizeResult]:
    yield optimize_parameters(
        partial(svm.NuSVC, kernel="linear", probability=False),
        "SVM, linear",
        lambda df: preprocessing.StandardScaler().fit_transform(df),
        nu=linspace(0.33903, 0.33905, 1000),
    )

    yield optimize_parameters(
        partial(svm.NuSVC, kernel="poly", degree=2, probability=False),
        "SVM, polynomial (quadratic)",
        lambda df: preprocessing.StandardScaler().fit_transform(df),
        nu=linspace(0.0004, 0.0005, 50),
        coef0=linspace(0.55, 0.65, 50)
    )

    yield optimize_parameters(
        partial(svm.NuSVC, kernel="poly", degree=3, probability=False),
        "SVM, polynomial (cubic)",
        lambda df: preprocessing.StandardScaler().fit_transform(df),
        nu=linspace(0.3215, 0.3225, 50),
        coef0=linspace(21.1, 21.3, 50)
    )

    yield optimize_parameters(
        partial(svm.NuSVC, kernel="poly", degree=4, probability=False),
        "SVM, polynomial (quartic)",
        lambda df: preprocessing.StandardScaler().fit_transform(df),
        nu=linspace(8e-9, 8.5e-9, 50),
        coef0=linspace(35, 35.1, 50)
    )

    yield optimize_parameters(
        partial(svm.NuSVC, kernel="rbf", probability=False),
        "SVM, RBF",
        lambda df: preprocessing.StandardScaler().fit_transform(df),
        nu=linspace(0.000825, 0.000827, 1000)
    )

    yield optimize_parameters(
        partial(svm.NuSVC, kernel="sigmoid", probability=False),
        "SVM, sigmoid",
        lambda df: preprocessing.StandardScaler().fit_transform(df),
        nu=linspace(0.40753, 0.407754, 1000)
    )


def test_linear_model() -> Iterator[OptimizeResult]:
    yield optimize_parameters(
        partial(linear_model.PassiveAggressiveClassifier, random_state=SEED, max_iter=2500),
        "Linear Model (passive aggressive)",
        C=linspace(0.00324, 0.00344, 100)
    )

    yield optimize_parameters(
        partial(linear_model.RidgeClassifier, random_state=SEED, max_iter=2500, solver="lsqr"),
        "Linear Model (ridge)",
        alpha=linspace(0.0007074, 0.0007275, 100)
    )

    # squared_error and squared_epsilon_insensitive do not converge in any reasonable number of iterations.
    yield optimize_parameters(
        partial(linear_model.SGDClassifier, random_state=SEED, max_iter=2500, loss="squared_hinge"),
        "Linear Model (passive aggressive)",
        alpha=linspace(0.06265, 0.6985, 100)
    )


def test_naive_bayes() -> Iterator[OptimizeResult]:
    yield optimize_parameters(
        naive_bayes.BernoulliNB,
        "Naive Bayes (Bernoulli)",
        alpha=linspace(0.05, 0.0501, 1000)
    )

    # Except for the Bernoulli model, Naive Bayes does not work with negative values.


def test_neighbors() -> Iterator[OptimizeResult]:
    yield optimize_parameters(
        partial(neighbors.KNeighborsClassifier, weights="distance"),
        "k Nearest Neighbors",
        n_neighbors=(1,)
    )

    yield optimize_parameters(
        partial(neighbors.RadiusNeighborsClassifier, weights="distance"),
        "Radius Neighbors",
        radius=linspace(23.7, 24.6, 50)
    )


def test_tree() -> Iterator[OptimizeResult]:
    yield optimize_parameters(
        partial(tree.DecisionTreeClassifier, max_features=240),
        "Decision Tree",
        criterion=("gini", "entropy", "log_loss"),
        splitter=("best", "random"),
        max_depth=range(1, 100, 10),
        min_samples_split=range(1, 10, 5),
        min_samples_leaf=range(1, 10, 5)
    )

    yield optimize_parameters(
        partial(tree.ExtraTreeClassifier, max_features=240, random_state=SEED),
        "Extra Tree",
        criterion=("gini", "entropy", "log_loss"),
        splitter=("random", "best"),
        max_depth=range(1, 100, 10),
        min_samples_split=range(1, 10, 5),
        min_samples_leaf=range(1, 10, 5)
    )


def test_neural_network() -> Iterator[OptimizeResult]:
    # lbfgs. sgd, identity and relu do not converge in any reasonable number of iterations.
    yield optimize_parameters(
        # Even with this amount of iterations, sometimes warnings are raised.
        # This is difficult to avoid without compromising accuracy.
        partial(neural_network.MLPClassifier, max_iter=2500),
        "Neural Network (MLP)",
        # FIXME: type error
        # TODO: change these ranges
        hidden_layer_sizes=tuple(full(n, k) for n in range(1, 5) for k in range(1, 300, 30)),
        activation=("logistic", "tanh"),
        alpha=linspace(0.001, 10, 10)
    )


def test_ensemble() -> Iterator[OptimizeResult]:
    yield optimize_parameters(
        partial(ensemble.ExtraTreesClassifier, max_features=240, random_state=SEED),
        "Extra Trees",
        criterion=("gini", "entropy", "log_loss"),
        max_depth=range(1, 100, 10),
        min_samples_split=range(1, 10, 5),
        min_samples_leaf=range(1, 10, 5)
    )

    yield optimize_parameters(
        partial(ensemble.RandomForestClassifier, max_features=240, random_state=SEED),
        "Extra Trees",
        criterion=("gini", "entropy", "log_loss"),
        max_depth=range(1, 100, 10),
        min_samples_split=range(1, 50, 5),
        min_samples_leaf=range(1, 10, 5)
    )

    yield optimize_parameters(
        partial(ensemble.GradientBoostingClassifier, max_features=240),
        "Gradient Boosting",
        loss=("log_loss", "exponential"),
        n_estimators=range(1, 500, 50),
        subsample=linspace(0.1, 1, 5),
        criterion=("friedman_mse", "squared_error"),
        min_samples_split=range(1, 10, 5),
        min_samples_leaf=range(1, 10, 5)
    )

    yield optimize_parameters(
        partial(ensemble.HistGradientBoostingClassifier, max_features=240),
        "Histogram-based Gradient Boosting",
        n_estimators=range(1, 500, 50),
        subsample=linspace(0.1, 1, 5),
        criterion=("friedman_mse", "squared_error"),
        min_samples_split=range(1, 10, 5),
        min_samples_leaf=range(1, 10, 5)
    )