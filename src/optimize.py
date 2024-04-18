import sys
from dataclasses import dataclass
from functools import reduce
from itertools import product
from operator import mul
from time import time
from typing import Callable, Dict, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from pandas import DataFrame
from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split as split

from data import *

SEED = 1
# Leave as None to use all available threads.
MAX_THREADS = None


@dataclass
class OptimizeResult:
    """
    A wrapper class for the results of an optimization.
    """

    model_name: str
    accuracy_training: float
    accuracy_testing: float
    average_time: float
    best_params: Dict[str, float]

    def __str__(self):
        return f"""
Model: {self.model_name}
Accuracy on training split: {(self.accuracy_training * 100):.1f}
Accuracy on testing data: {(self.accuracy_testing * 100):.1f}
Average runtime: {self.average_time:.1e} seconds
Best parameters found:
{'\n'.join(f"{k} = {v}" for k, v in self.best_params.items())}
"""


# noinspection PyUnresolvedReferences,PyArgumentList
def optimize_parameters(model: Callable[..., ClassifierMixin],
                        name: str,
                        preprocessor: Callable[[DataFrame], DataFrame] = lambda df: df,
                        **params: Sequence
                        ) -> OptimizeResult:
    """
    Attempts to find the optimal parameters for a model by trying all combinations
    in given ranges. This operation can be very expensive.
    :param model: the constructor for the classifier to use.
    :param preprocessor: the preprocessing function to apply to the data before training, if any.
    :param name: the name of the model.
    :param params: all values to try for all the parameters to vary.
    For numeric paramters, these should likely be the results of a call to `numpy.linspace` or `range`.
    :return: the results of the optimization. See `OptimizeResult`.
    """

    def name_params(values):
        return dict(zip(names, values))

    # noinspection PyUnresolvedReferences,PyArgumentList
    def run(**config):
        try:
            classifier = model(**config)
        except TypeError:
            raise TypeError("Error: invalid parameters to model")

        try:
            classifier.fit(training_split, training_targets_split)
            return accuracy_score(verification_targets, classifier.predict(verification))
        except ValueError as e:
            with lock:
                print(f"Error: {e}\nModel: {classifier}", file=sys.stderr)
            return 0

    training = preprocessor(get_training_features())
    testing = preprocessor(get_testing_features())
    targets = get_training_targets()
    testing_targets = get_testing_targets()

    # 75% of training set used for training, 25% used to test accuracy (verification).
    training_split, verification, training_targets_split, verification_targets = split(
        training,
        targets,
        test_size=0.25,
        random_state=SEED
    )

    names = tuple(params.keys())
    iterations = reduce(mul, (len(p) for p in params.values()), 1)

    # Train model with the training split
    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        lock = Lock()
        start_time = time()

        futures = (executor.submit(lambda p: (p, run(**name_params(p))), p)
                   for p in product(*params.values()))
        results = tuple(future.result() for future in as_completed(futures))

        end_time = time()

    best_parameters, accuracy_training = max(results, key=lambda b: b[1])
    best_parameters = name_params(best_parameters)

    # Test model with entire training set and best found parameters
    classifier_testing = model(**best_parameters)
    classifier_testing.fit(training, targets)
    accuracy_testing = accuracy_score(testing_targets, classifier_testing.predict(testing))

    return OptimizeResult(name,
                          accuracy_training,
                          accuracy_testing,
                          (end_time - start_time) / iterations,
                          best_parameters)
