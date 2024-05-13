from random import randint
from typing import Tuple

from numpy import linspace
from sklearn import *

from optimize import optimize_parameters, OptimizeResult, SEED


class Just:
    """
    Represents an iterator yielding just a single value.
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


# These tests are for the final iteration of each model. For older tests, see the commit history on the respository.


def svm_linear() -> OptimizeResult:
    return optimize_parameters(
        svm.NuSVC,
        "SVM, linear",
        kernel=Just("linear"),
        probability=Just(False),
        nu=linspace(0.33903, 0.33905, 500)
    )


def svm_quad() -> OptimizeResult:
    return optimize_parameters(
        svm.NuSVC,
        "SVM, polynomial (quadratic)",
        kernel=Just("poly"),
        degree=Just(2),
        probability=Just(False),
        nu=linspace(0.0004, 0.0005, 25),
        coef0=linspace(0.55, 0.65, 25)
    )


def svm_cub() -> OptimizeResult:
    return optimize_parameters(
        svm.NuSVC,
        "SVM, polynomial (cubic)",
        kernel=Just("poly"),
        degree=Just(3),
        probability=Just(False),
        nu=linspace(0.3215, 0.3225, 25),
        coef0=linspace(21.1, 21.3, 25)
    )


def svm_quar() -> OptimizeResult:
    return optimize_parameters(
        svm.NuSVC,
        "SVM, polynomial (quartic)",
        kernel=Just("poly"),
        degree=Just(4),
        probability=Just(False),
        nu=linspace(8e-9, 8.5e-9, 25),
        coef0=linspace(35, 35.1, 25)
    )


def svm_rbf() -> OptimizeResult:
    return optimize_parameters(
        svm.NuSVC,
        "SVM, RBF",
        kernel=Just("rbf"),
        probability=Just(False),
        nu=linspace(0.000825, 0.000827, 500)
    )


def svm_sigmoid() -> OptimizeResult:
    return optimize_parameters(
        svm.NuSVC,
        "SVM, sigmoid",
        kernel=Just("sigmoid"),
        probability=Just(False),
        nu=linspace(0.40753, 0.407754, 500)
    )


def linear_model_pa() -> OptimizeResult:
    return optimize_parameters(
        linear_model.PassiveAggressiveClassifier,
        "Linear Model (passive aggressive)",
        random_state=Just(SEED),
        max_iter=Just(2500),
        C=linspace(0.00324, 0.00344, 50)
    )


def linear_model_ridge() -> OptimizeResult:
    return optimize_parameters(
        linear_model.RidgeClassifier,
        "Linear Model (ridge)",
        random_state=Just(SEED),
        max_iter=Just(2500),
        solver=Just("lsqr"),
        alpha=linspace(0.0007074, 0.0007275, 50)
    )


def linear_model_sgd() -> OptimizeResult:
    # squared_error and squared_epsilon_insensitive do not converge in any reasonable number of iterations.
    return optimize_parameters(
        linear_model.SGDClassifier,
        "Linear Model (stochastic gradient descent)",
        random_state=Just(SEED),
        max_iter=Just(2500),
        loss=Just("squared_hinge"),
        alpha=linspace(0.06265, 0.6985, 50)
    )


def naive_bayes() -> OptimizeResult:
    return optimize_parameters(
        naive_bayes.BernoulliNB,
        "Naive Bayes (Bernoulli)",
        alpha=linspace(0.05, 0.0501, 500)
    )


def knn() -> OptimizeResult:
    return optimize_parameters(
        neighbors.KNeighborsClassifier,
        "k Nearest Neighbors",
        weights=Just("distance"),
        n_neighbors=Just(1)
    )


def radius_neighbors() -> OptimizeResult:
    return optimize_parameters(
        neighbors.RadiusNeighborsClassifier,
        "Radius Neighbors",
        weights=Just("distance"),
        radius=linspace(23.7, 24.6, 50)
    )


def decision_tree() -> OptimizeResult:
    return optimize_parameters(
        tree.DecisionTreeClassifier,
        "Decision Tree",
        max_features=Just(240),
        criterion=Just("log_loss"),
        splitter=Just("best"),
        min_samples_leaf=Just(3),
        max_depth=Just(28),
        min_samples_split=linspace(0.0002, 0.0003, 500)
    )


def extra_tree() -> OptimizeResult:
    return optimize_parameters(
        tree.ExtraTreeClassifier,
        "Extra Tree",
        max_features=Just(240),
        random_state=Just(SEED),
        criterion=Just("entropy"),
        splitter=Just("best"),
        min_samples_leaf=Just(2),
        max_depth=Just(7),
        min_samples_split=linspace(1e-12, 1e-9, 500)
    )


def nn_equal() -> OptimizeResult:
    return optimize_parameters(
        neural_network.MLPClassifier,
        "Multilayer Perceptron (all hidden layers equal size)",
        max_iter=Just(2000),
        random_state=Just(SEED),
        activation=Just("tanh"),
        hidden_layer_sizes=Just((159, 159, 159)),
        alpha=linspace(1e-6, 1e-5, 10)
    )


def nn_single() -> OptimizeResult:
    return optimize_parameters(
        neural_network.MLPClassifier,
        "Multilayer Perceptron (single hidden layer)",
        max_iter=Just(2000),
        random_state=Just(SEED),
        activation=Just("tanh"),
        hidden_layer_sizes=Just((161,)),
        alpha=linspace(2.341e-13, 4.339e-13, 10)
    )


def nn_random() -> OptimizeResult:
    def random_layers(min_layers, max_layers, min_neurons, max_neurons) -> Tuple[int, ...]:
        return tuple(randint(min_neurons, max_neurons) for _ in range(randint(min_layers, max_layers)))

    return optimize_parameters(
        neural_network.MLPClassifier,
        "Multilayer Perceptron (randomly sampled layouts)",
        max_iter=Just(2000),
        random_state=Just(SEED),
        activation=Just("tanh"),
        alpha=Just(1e-7),
        hidden_layer_sizes=Just((random_layers(1, 3, 100, 300) for _ in range(10)))
    )


def extra_trees() -> OptimizeResult:
    return optimize_parameters(
        ensemble.ExtraTreesClassifier,
        "Extra Trees",
        max_features=Just(240),
        random_state=Just(SEED),
        criterion=Just("entropy"),
        max_depth=Just(10),
        min_samples_split=linspace(4.14e-8, 6.15e-8, 50),
    )


def random_forest() -> OptimizeResult:
    return optimize_parameters(
        ensemble.RandomForestClassifier,
        "Random Forest",
        max_features=Just(240),
        random_state=Just(SEED),
        criterion=Just("gini"),
        max_depth=Just(10),
        min_samples_split=linspace(3.13e-8, 5.14e-8, 50),
    )


def grad_boost() -> OptimizeResult:
    return optimize_parameters(
        ensemble.HistGradientBoostingClassifier,
        "Histogram-based Gradient Boosting",
        max_leaf_nodes=range(2, 11, 2),
        learning_rate=linspace(0.1, 1, 5),
        min_samples_leaf=range(1, 100, 25)
    )


def ada_boost() -> OptimizeResult:
    return optimize_parameters(
        ensemble.AdaBoostClassifier,
        "AdaBoost",
        random_state=Just(SEED),
        algorithm=Just("SAMME"),
        n_estimators=Just(427),
        learning_rate=linspace(5.93, 6.423, 50)
    )


def bagging() -> OptimizeResult:
    return optimize_parameters(
        ensemble.BaggingClassifier,
        "Bagging",
        max_features=Just(240),
        random_state=Just(SEED),
        n_estimators=Just(168),
        max_samples=linspace(0.489, 0.533, 50)
    )
