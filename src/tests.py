from functools import partial
from random import randint
from typing import Iterator

from numpy import linspace
from sklearn import *

from optimize import optimize_parameters, OptimizeResult, SEED


# These tests are for the final iteration of each model. For older tests, see the commit history on the respository.


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
        "Linear Model (stochastic gradient descent)",
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
        partial(tree.DecisionTreeClassifier, max_features=240,
                criterion="log_loss", splitter="best", min_samples_leaf=3, max_depth=28),
        "Decision Tree",
        min_samples_split=linspace(0.0002, 0.0003, 1000)
    )

    yield optimize_parameters(
        partial(tree.ExtraTreeClassifier, max_features=240, random_state=SEED,
                criterion="entropy", splitter="best", min_samples_leaf=2, max_depth=7),
        "Extra Tree",
        min_samples_split=linspace(1e-12, 1e-9, 1000)
    )


def test_neural_network() -> Iterator[OptimizeResult]:
    yield optimize_parameters(
        partial(neural_network.MLPClassifier, max_iter=2000, random_state=SEED,
                activation="tanh", hidden_layer_sizes=(159, 159, 159)),
        "Multilayer Perceptron (all hidden layers equal size)",
        alpha=linspace(1e-6, 1e-5, 10)
    )

    yield optimize_parameters(
        partial(neural_network.MLPClassifier, max_iter=2000, random_state=SEED,
                activation="tanh", hidden_layer_sizes=(161,)),
        "Multilayer Perceptron (single hidden layer)",
        alpha=linspace(2.341e-13, 4.339e-13, 10)
    )

    min_layers = 1
    max_layers = 3
    min_neurons = 100
    max_neurons = 300

    yield optimize_parameters(
        partial(neural_network.MLPClassifier, max_iter=2000, random_state=SEED,
                activation="tanh", alpha=1e-7),
        "Multilayer Perceptron (randomly sampled layouts)",
        hidden_layer_sizes=tuple(tuple(randint(min_neurons, max_neurons)
                                       for _ in range(randint(min_layers, max_layers)))
                                 for _ in range(100))
    )


def test_ensemble() -> Iterator[OptimizeResult]:
    yield optimize_parameters(
        partial(ensemble.ExtraTreesClassifier, max_features=240, random_state=SEED,
                criterion="entropy", max_depth=10),
        "Extra Trees",
        min_samples_split=linspace(4.14e-8, 6.15e-8, 50),
    )

    yield optimize_parameters(
        partial(ensemble.RandomForestClassifier, max_features=240, random_state=SEED,
                criterion="gini", max_depth=10),
        "Random Forest",
        min_samples_split=linspace(3.13e-8, 5.14e-8, 50),
    )

    # GradientBoostingClassifier omitted due to performance intensity on larger datasets.

    yield optimize_parameters(
        partial(ensemble.HistGradientBoostingClassifier),
        "Histogram-based Gradient Boosting",
        max_leaf_nodes=range(2, 11, 2),
        learning_rate=linspace(0.1, 1, 5),
        min_samples_leaf=range(1, 100, 20)
    )

    yield optimize_parameters(
        partial(ensemble.AdaBoostClassifier, random_state=SEED, algorithm="SAMME", n_estimators=427),
        "AdaBoost",
        learning_rate=linspace(5.93, 6.423, 100)
    )

    yield optimize_parameters(
        partial(ensemble.BaggingClassifier, max_features=240, random_state=SEED, n_estimators=168),
        "Bagging",
        max_samples=linspace(0.489, 0.533, 100)
    )
