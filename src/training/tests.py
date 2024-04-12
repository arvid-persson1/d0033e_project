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
