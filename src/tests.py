from random import randint
from typing import Tuple

from numpy import linspace, logspace
from sklearn import *

from optimize import optimize, OptimizeResult


class Just:
    """
    Represents an iterator yielding just a single value.
    More explicit than constructing 1-length tuples.
    """

    def __init__(self, value):
        self.value = value
        self.yielded = False

    def __iter__(self):
        return self

    def __next__(self):
        if not self.yielded:
            self.yielded = True
            return self.value
        else:
            raise StopIteration


def svm_linear() -> OptimizeResult:
    return optimize(
        svm.NuSVC,
        "SVM, linear",
        scale_data=True,
        kernel=Just("linear"),
        probability=Just(False),
        nu=Just(0.3390397404411425)
    )


def svm_quad() -> OptimizeResult:
    return optimize(
        svm.NuSVC,
        "SVM, polynomial (quadratic)",
        scale_data=True,
        kernel=Just("poly"),
        degree=Just(2),
        probability=Just(False),
        nu=linspace(0.3194444444444451, 0.32638888888888956, 50),
        coef0=linspace(10.41666666666667, 11.805555555555559, 50)
    )


def svm_cub() -> OptimizeResult:
    return optimize(
        svm.NuSVC,
        "SVM, polynomial (cubic)",
        scale_data=True,
        kernel=Just("poly"),
        degree=Just(3),
        probability=Just(False),
        nu=linspace(1e-15, 0.08333333333333424, 50),
        coef0=linspace(16.66666666666667, 33.33333333333334, 50)
    )


def svm_quar() -> OptimizeResult:
    return optimize(
        svm.NuSVC,
        "SVM, polynomial (quartic)",
        scale_data=True,
        kernel=Just("poly"),
        degree=Just(4),
        probability=Just(False),
        nu=linspace(0.3222789115646265, 0.3256802721088442, 50),
        coef0=linspace(30.612244897959158, 37.41496598639453, 50)
    )


def svm_rbf() -> OptimizeResult:
    return optimize(
        svm.NuSVC,
        "SVM, RBF",
        scale_data=True,
        kernel=Just("rbf"),
        probability=Just(False),
        nu=linspace(0.01010101010101109, 0.030303030303031275, 1000)
    )


def svm_sigmoid() -> OptimizeResult:
    return optimize(
        svm.NuSVC,
        "SVM, sigmoid",
        scale_data=True,
        kernel=Just("sigmoid"),
        probability=Just(False),
        nu=linspace(0.3939393939393946, 0.41414141414141475, 1000),
        coef0=Just(0)
    )


def linear_model_pa() -> OptimizeResult:
    return optimize(
        linear_model.PassiveAggressiveClassifier,
        "Linear Model (passive aggressive)",
        scale_data=True,
        num_trials=5,
        C=logspace(5.727272727272727, 5.828282828282829, 100)
    )


def linear_model_ridge() -> OptimizeResult:
    return optimize(
        linear_model.RidgeClassifier,
        "Linear Model (ridge)",
        scale_data=True,
        num_trials=5,
        alpha=logspace(-2.25, -0.75, 100),
        solver=Just("auto")
    )


def linear_model_sgd() -> OptimizeResult:
    return optimize(
        linear_model.SGDClassifier,
        "Linear Model (stochastic gradient descent)",
        scale_data=True,
        num_trials=5,
        alpha=logspace(-4.428571428571429, -4.26530612244898, 50),
        loss=Just("squared_hinge"),
        penalty=Just("l2")
    )


def naive_bayes() -> OptimizeResult:
    return optimize(
        naive_bayes.BernoulliNB,
        "Naive Bayes (Bernoulli)",
        alpha=logspace(-15, 3, 1000)
    )


def knn() -> OptimizeResult:
    return optimize(
        neighbors.KNeighborsClassifier,
        "k Nearest Neighbors",
        scale_data=True,
        n_neighbors=range(1, 101),
        weights=("uniform", "distance")
    )


def radius_neighbors() -> OptimizeResult:
    return optimize(
        neighbors.RadiusNeighborsClassifier,
        "Radius Neighbors",
        scale_data=True,
        radius=logspace(-15, 3, 500),
        weights=("uniform", "distance")
    )


def decision_tree() -> OptimizeResult:
    return optimize(
        tree.DecisionTreeClassifier,
        "Decision Tree",
        num_trials=5,
        criterion=("gini", "entropy", "log_loss"),
        splitter=("best", "random"),
        min_samples_split=linspace(1e-15, 1 - 1e-15, 5),
        min_samples_leaf=linspace(1e-15, 1 - 1e-15, 5),
        max_features=("log2", "sqrt", None)
    )


def extra_tree() -> OptimizeResult:
    return optimize(
        tree.ExtraTreeClassifier,
        "Extra Tree",
        num_trials=5,
        criterion=("gini", "entropy", "log_loss"),
        splitter=("best", "random"),
        min_samples_split=linspace(1e-15, 1 - 1e-15, 5),
        min_samples_leaf=linspace(1e-15, 1 - 1e-15, 5),
        max_features=("log2", "sqrt", None)
    )


def nn_equal() -> OptimizeResult:
    def equal_layers(count: int, size: int) -> Tuple[int, ...]:
        return tuple(size for _ in range(count))

    return optimize(
        neural_network.MLPClassifier,
        "Multilayer Perceptron (all hidden layers equal size)",
        scale_data=True,
        num_trials=5,
        hidden_layer_sizes=(equal_layers(count, size) for count in range(1, 11, 3) for size in range(1, 502, 100)),
        activation=("identity", "logistic", "tanh", "relu"),
        alpha=logspace(-15, 3, 5)
    )


def nn_single() -> OptimizeResult:
    def single_layer(size: int) -> Tuple[int]:
        return (size,)

    return optimize(
        neural_network.MLPClassifier,
        "Multilayer Perceptron (single hidden layer)",
        scale_data=True,
        num_trials=5,
        hidden_layer_sizes=(single_layer(size) for size in range(1, 502, 50)),
        activation=("identity", "logistic", "tanh", "relu"),
        alpha=logspace(-15, 3, 5)
    )


def nn_random() -> OptimizeResult:
    def random_layers(min_layers: int, max_layers: int, min_neurons: int, max_neurons: int) -> Tuple[int, ...]:
        return tuple(randint(min_neurons, max_neurons) for _ in range(randint(min_layers, max_layers)))

    return optimize(
        neural_network.MLPClassifier,
        "Multilayer Perceptron (randomly sampled layouts)",
        scale_data=True,
        max_iter=Just(2000),
        num_trials=5,
        hidden_layer_sizes=(random_layers(1, 10, 1, 500) for _ in range(50)),
        activation=("identity", "logistic", "tanh", "relu"),
        alpha=logspace(-15, 3, 5)
    )


def random_forest() -> OptimizeResult:
    return optimize(
        ensemble.RandomForestClassifier,
        "Random Forest",
        num_trials=5,
        criterion=("gini", "entropy", "log_loss"),
        min_samples_split=linspace(1e-15, 1 - 1e-15, 5),
        min_samples_leaf=linspace(1e-15, 1 - 1e-15, 5),
        max_features=("log2", "sqrt", None),
        bootstrap=(True, False)
    )


def extra_trees() -> OptimizeResult:
    return optimize(
        ensemble.ExtraTreesClassifier,
        "Extra Trees",
        num_trials=5,
        criterion=("gini", "entropy", "log_loss"),
        min_samples_split=linspace(1e-15, 1 - 1e-15, 5),
        min_samples_leaf=linspace(1e-15, 1 - 1e-15, 5),
        max_features=("log2", "sqrt", None),
        bootstrap=(True, False)
    )


def grad_boost() -> OptimizeResult:
    return optimize(
        ensemble.HistGradientBoostingClassifier,
        "Histogram-based Gradient Boosting",
        max_leaf_nodes=Just(None),
        learning_rate=linspace(0, 1, 5),
        min_samples_leaf=range(1, 102, 10)
    )


def ada_boost() -> OptimizeResult:
    return optimize(
        ensemble.AdaBoostClassifier,
        "AdaBoost (decision stumps)",
        num_trials=5,
        algorithm=Just("SAMME"),
        n_estimators=range(1, 502, 50),
        learning_rate=logspace(-15, 3, 10)
    )


def bagging() -> OptimizeResult:
    return optimize(
        ensemble.BaggingClassifier,
        "Bagging",
        num_trials=5,
        n_estimators=range(1, 502, 50),
        max_samples=range(1, 541, 10)
    )
