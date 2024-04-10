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
