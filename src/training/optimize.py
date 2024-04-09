from typing import Optional, Callable, Dict, Iterable, Sequence
from functools import reduce
from operator import mul
from time import time
from itertools import product
from dataclasses import dataclass

from pandas import DataFrame, Series
import sklearn
from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score

from src.data import *

sklearn.set_config(assume_finite=True)


def accuracy(model: Callable[..., ClassifierMixin],
             preprocessor: Optional[Callable[[DataFrame], DataFrame]] = None,
             **params
             ) -> float:
    """
    Tests the accuracy of a model.
    :param model: the constructor for the classifier to use.
    :param preprocessor: the preprocessing function to apply to the data before training, if any.
    :param params: parameters to pass to the model.
    :return: the accuracy on the testing data. Will always be between 0 and 1.
    """

    training_features = get_training_features()
    testing_features = get_testing_features()
    training_targets = get_training_targets()
    testing_targets = get_testing_targets()

    if preprocessor is not None:
        training_features = preprocessor(training_features)
        testing_features = preprocessor(testing_features)

    return __run_model(model,
                       training_features,
                       training_targets,
                       testing_features,
                       testing_targets,
                       **params)


# Python type annotations are not sophisticated enough to typecheck this function properly.
# noinspection PyUnresolvedReferences,PyArgumentList
def __run_model(model: Callable[..., ClassifierMixin],
                training_features: DataFrame,
                training_targets: Series,
                testing_features: DataFrame,
                testing_targets: Series,
                **params
                ) -> float:
    try:
        classifier = model(**params)
    except TypeError:
        raise TypeError("invalid parameters to model")

    try:
        classifier.fit(training_features, training_targets)
        prediction = classifier.predict(testing_features)
    except ValueError:
        # TODO: log errors
        return 0

    result = accuracy_score(testing_targets, prediction)

    return result


@dataclass
class OptimizeResult:
    """
    A wrapper for the results of an optimization.

    `params` contains the names of parameters and the best value found for them.

    `accuracy` returns the accuracy of the model given `params`.
    """

    params: Dict[str, float]
    accuracy: float
    average_time: float

    def display(self, name: str):
        """
        Prints the results of the optimization.
        :param name: the name of the model.
        """

        print(name)
        print(f"Accuracy: {(self.accuracy * 100):.2f}%")

        print("Best parameters:")
        for k, v in self.params.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.3f}")
            else:
                print(f"  {k}: '{v}'")

        print(f"Average time: {self.average_time:.2e}s\n")


def optimize_parameters(model: Callable[..., ClassifierMixin],
                        preprocessor: Optional[Callable[[DataFrame], DataFrame]] = None,
                        **params: Sequence
                        ) -> OptimizeResult:
    """
    Attempts to find the optimal parameters for a model by trying all combinations
    in given ranges. Warning: this operation can be very expensive.
    :param model: the constructor for the classifier to use.
    :param preprocessor: the preprocessing function to apply to the data before training, if any.
    :param params: all values to try for all the parameters to vary.
    For numeric paramters, these should likely be the results of a call to `numpy.linspace`.
    :return: the results of the optimization. See `OptimizeResult`.
    """

    # This repeat logic shouldn't be moved to __run_model since it would
    # cause repeated calls to the potentially expensive preprocessor.
    training_features = get_training_features()
    training_targets = get_training_targets()
    testing_features = get_testing_features()
    testing_targets = get_testing_targets()

    if preprocessor is not None:
        training_features = preprocessor(training_features)
        testing_features = preprocessor(testing_features)

    names = tuple(params.keys())

    def bundle(values: Iterable[float]) -> Dict[str, float]:
        return dict(zip(names, values))

    iterations = reduce(mul, (len(p) for p in params.values()), 1)
    start_time = time()

    best_config, best_accuracy = max(
        ((cfg, __run_model(model,
                           training_features,
                           training_targets,
                           testing_features,
                           testing_targets,
                           **bundle(cfg)))
         for cfg in product(*params.values())),
        key=lambda best: best[1]
    )

    end_time = time()

    return OptimizeResult(bundle(best_config), best_accuracy, (end_time - start_time) / iterations)
