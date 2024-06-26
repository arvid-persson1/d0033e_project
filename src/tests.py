from functools import partial
from random import randint
from typing import Tuple

from numpy import linspace, logspace
from sklearn import *
from sklearn.tree import DecisionTreeClassifier

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
        preprocessor="scale",
        kernel=Just("linear"),
        probability=Just(False),
        nu=Just(0.3390397404411425)
    )


def svm_quad() -> OptimizeResult:
    return optimize(
        svm.NuSVC,
        "SVM, polynomial (quadratic)",
        preprocessor="scale",
        kernel=Just("poly"),
        degree=Just(2),
        probability=Just(False),
        nu=Just(0.3218537414965993),
        coef0=Just(10.41666666666667)
    )


def svm_cub() -> OptimizeResult:
    return optimize(
        svm.NuSVC,
        "SVM, polynomial (cubic)",
        preprocessor="scale",
        kernel=Just("poly"),
        degree=Just(3),
        probability=Just(False),
        nu=Just(0.0017006802721098418),
        coef0=Just(20.7482993197279)
    )


def svm_quar() -> OptimizeResult:
    return optimize(
        svm.NuSVC,
        "SVM, polynomial (quartic)",
        preprocessor="scale",
        kernel=Just("poly"),
        degree=Just(4),
        probability=Just(False),
        nu=Just(0.3222789115646265),
        coef0=Just(32.27821740941272)
    )


def svm_rbf() -> OptimizeResult:
    return optimize(
        svm.NuSVC,
        "SVM, RBF",
        preprocessor="scale",
        kernel=Just("rbf"),
        probability=Just(False),
        nu=Just(0.012467012467013455)
    )


def svm_sigmoid() -> OptimizeResult:
    return optimize(
        svm.NuSVC,
        "SVM, sigmoid",
        preprocessor="scale",
        kernel=Just("sigmoid"),
        probability=Just(False),
        nu=Just(0.4075894075894082),
        coef0=Just(0)
    )


def linear_model_pa() -> OptimizeResult:
    return optimize(
        linear_model.PassiveAggressiveClassifier,
        "Linear Model (passive aggressive)",
        preprocessor="scale",
        num_trials=5,
        C=Just(582135.3562373725)
    )


def linear_model_ridge() -> OptimizeResult:
    return optimize(
        linear_model.RidgeClassifier,
        "Linear Model (ridge)",
        preprocessor="scale",
        num_trials=5,
        alpha=Just(0.03217923812785397),
        solver=Just("auto")
    )


def linear_model_sgd() -> OptimizeResult:
    return optimize(
        linear_model.SGDClassifier,
        "Linear Model (stochastic gradient descent)",
        preprocessor="scale",
        num_trials=5,
        alpha=Just(5.3871855299887975e-05),
        loss=Just("squared_hinge"),
        penalty=Just("l2")
    )


def bayes() -> OptimizeResult:
    return optimize(
        naive_bayes.BernoulliNB,
        "Naive Bayes (Bernoulli)",
        alpha=Just(0.016114142772530166)
    )


def knn() -> OptimizeResult:
    return optimize(
        neighbors.KNeighborsClassifier,
        "k Nearest Neighbors",
        preprocessor="scale",
        n_neighbors=Just(1),
        weights=Just("uniform")
    )


def radius_neighbors() -> OptimizeResult:
    return optimize(
        neighbors.RadiusNeighborsClassifier,
        "Radius Neighbors",
        preprocessor="scale",
        radius=Just(58.611111111111114),
        weights=Just("distance")
    )


def decision_tree() -> OptimizeResult:
    return optimize(
        tree.DecisionTreeClassifier,
        "Decision Tree",
        num_trials=5,
        criterion=Just("entropy"),
        splitter=Just("random"),
        min_samples_split=Just(1e-18),
        min_samples_leaf=Just(0.0022675736961451413),
        max_features=Just(None)
    )


def extra_tree() -> OptimizeResult:
    return optimize(
        tree.ExtraTreeClassifier,
        "Extra Tree",
        num_trials=5,
        criterion=Just("entropy"),
        splitter=Just("best"),
        min_samples_split=Just(0.018140589569160984),
        min_samples_leaf=Just(1e-18),
        max_features=Just(None)
    )


def nn_equal() -> OptimizeResult:
    def equal_layers(count: int, size: int) -> Tuple[int, ...]:
        return tuple(size for _ in range(count))

    return optimize(
        neural_network.MLPClassifier,
        "Multilayer Perceptron (all hidden layers equal size)",
        preprocessor="scale",
        num_trials=25,
        max_iter=Just(4000),
        hidden_layer_sizes=Just(equal_layers(2, 161)),
        activation=Just("logistic"),
        alpha=Just(2.371373705661655e-10)
    )


def nn_single() -> OptimizeResult:
    def single_layer(size: int) -> Tuple[int]:
        return (size,)

    return optimize(
        neural_network.MLPClassifier,
        "Multilayer Perceptron (single hidden layer)",
        preprocessor="scale",
        num_trials=25,
        max_iter=Just(2500),
        hidden_layer_sizes=Just(single_layer(150)),
        activation=Just("logistic"),
        alpha=Just(1e-15)
    )


def nn_random() -> OptimizeResult:
    def random_layers(min_layers: int, max_layers: int, min_neurons: int, max_neurons: int) -> Tuple[int, ...]:
        return tuple(randint(min_neurons, max_neurons) for _ in range(randint(min_layers, max_layers)))

    return optimize(
        neural_network.MLPClassifier,
        "Multilayer Perceptron (randomly sampled layouts)",
        preprocessor="scale",
        num_trials=15,
        max_iter=Just(2500),
        hidden_layer_sizes=(random_layers(1, 2, 120, 200) for _ in range(50)),
        activation=Just("logistic"),
        alpha=logspace(-15, -12, 5)
    )


def random_forest() -> OptimizeResult:
    return optimize(
        ensemble.RandomForestClassifier,
        "Random Forest",
        num_trials=25,
        criterion=Just("log_loss"),
        min_samples_split=linspace(1e-24, 0.013888888888888888, 100),
        min_samples_leaf=Just(1e-24),
        max_features=Just("sqrt"),
        bootstrap=Just(False)
    )


def extra_trees() -> OptimizeResult:
    return optimize(
        ensemble.ExtraTreesClassifier,
        "Extra Trees",
        num_trials=1000,
        criterion=Just("log_loss"),
        min_samples_split=Just(3e-3),
        min_samples_leaf=Just(1e-3),
        max_features=Just(None),
        bootstrap=Just(False)
    )


def grad_boost() -> OptimizeResult:
    return optimize(
        ensemble.HistGradientBoostingClassifier,
        "Histogram-based Gradient Boosting",
        max_leaf_nodes=Just(100),
        learning_rate=Just(0.11547819846894582),
        min_samples_leaf=Just(18)
    )


def ada_boost() -> OptimizeResult:
    return optimize(
        ensemble.AdaBoostClassifier,
        "AdaBoost (decision stumps)",
        num_trials=50,
        algorithm=Just("SAMME"),
        n_estimators=Just(500),
        learning_rate=Just(6.080658554525504)
    )


def bagging_default() -> OptimizeResult:
    return optimize(
        ensemble.BaggingClassifier,
        "Bagging (default classifier)",
        num_trials=25,
        n_estimators=Just(500),
        max_samples=Just(0.5)
    )


def bagging_custom() -> OptimizeResult:
    return optimize(
        partial(
            # Optimized parameters for decision tree
            ensemble.BaggingClassifier,
            estimator=Just(DecisionTreeClassifier(
                criterion="entropy", splitter="random", min_samples_split=1e-18,
                min_samples_leaf=0.0022675736961451413, max_features=None
            ))
        ),
        "Bagging (default classifier)",
        num_trials=25,
        n_estimators=Just(500),
        max_samples=Just(0.4)
    )


def svm_linear_pca() -> OptimizeResult:
    return optimize(
        svm.NuSVC,
        "SVM, linear, PCA",
        preprocessor="pca",
        kernel=Just("linear"),
        probability=Just(False),
        nu=Just(4.026230252556475e-05)
    )


def svm_quad_pca() -> OptimizeResult:
    return optimize(
        svm.NuSVC,
        "SVM, polynomial (quadratic), PCA",
        preprocessor="pca",
        kernel=Just("poly"),
        degree=Just(2),
        probability=Just(False),
        nu=Just(0.001959183673979592),
        coef0=Just(0.6111399161913893)
    )


def nn_single_pca() -> OptimizeResult:
    def single_layer(size: int) -> Tuple[int]:
        return (size,)

    return optimize(
        neural_network.MLPClassifier,
        "Multilayer Perceptron (single hidden layer), PCA",
        preprocessor="pca",
        num_trials=15,
        max_iter=Just(2500),
        hidden_layer_sizes=(single_layer(n) for n in range(51, 152, 25)),
        activation=Just("tanh"),
        alpha=logspace(-1.6666666666666667, -0.7777777777777777, 10)
    )


def random_forest_pca() -> OptimizeResult:
    return optimize(
        ensemble.RandomForestClassifier,
        "Random Forest, PCA",
        preprocessor="pca",
        num_trials=15,
        criterion=Just("gini"),
        min_samples_split=Just(1e-15),
        min_samples_leaf=Just(4.445000000000001e-16)
    )


def extra_trees_pca() -> OptimizeResult:
    return optimize(
        ensemble.ExtraTreesClassifier,
        "Extra Trees, PCA",
        preprocessor="pca",
        num_trials=25,
        criterion=Just("gini"),
        min_samples_split=Just(7.867566666666669e-16),
        min_samples_leaf=Just(1e-21)
    )
