from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import product
from sys import stderr
from threading import Lock
from time import time
from typing import Callable, Dict, Any, Iterable, Optional

from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split as split

from data import *

# Used for splitting the datasets.
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
    best_params: Dict[str, Any]

    def __str__(self):
        return f"""Model: {self.model_name}
Accuracy on verification split: {(self.accuracy_training * 100):.1f}%
Accuracy on testing data: {(self.accuracy_testing * 100):.1f}%
Average runtime: {self.average_time:.1e} seconds
Best parameters found:
{'\n'.join(f"{k} = {v}" for k, v in self.best_params.items())}
"""


# noinspection PyUnresolvedReferences,PyArgumentList
def optimize(model: Callable[..., ClassifierMixin], name: str, *, preprocessor: Optional[str] = None,
             num_trials: int = 1, **params: Iterable[Any]) -> OptimizeResult:
    """
    Attempts to find the optimal parameters for a model by trying all combinations
    in given ranges. This operation can be very expensive.
    :param model: the constructor for the classifier to use.
    :param name: the name of the model.
    :param preprocessor: how to process the data before training, if at all.
    `scale` uses scaled values. `pca` uses principal components.
    :param params: all values to try for all the parameters to vary.
    :param num_trials: the number of trials to run to account for random variation.
    The median result is returned.
    :return: the results of the optimization. See `OptimizeResult`.
    """

    def name_params(values: Iterable[Any]) -> dict[str, Any]:
        return dict(zip(params.keys(), values))

    # noinspection PyUnresolvedReferences,PyArgumentList
    def run(**config) -> float:
        try:
            classifier = model(**config)
        except TypeError as e:
            raise ValueError(f"Error: invalid parameters to model: {e}")

        try:
            classifier.fit(trn_ft, trn_tg)
            return accuracy_score(val_tg, classifier.predict(val_ft))
        except ValueError as e:
            with lock:
                print(f"Error: {e}\nModel: {classifier}\n", file=stderr)
            return 0

    match preprocessor:
        case "scale":
            trn_ft = training_features_scaled()
            tst_ft = testing_features_scaled()
        case "pca":
            trn_ft = training_pc()
            tst_ft = testing_pc()
        case None:
            trn_ft = training_features()
            tst_ft = testing_features()
        case _:
            raise ValueError("Error: invalid preprocessor.")

    trn_tg = training_targets()
    tst_tg = testing_targets()

    # 75% for training, 25% for validation.
    trn_ft, val_ft, trn_tg, val_tg = split(trn_ft, trn_tg, test_size=0.25, random_state=SEED)

    results = []

    # Train model with the training split
    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        # noinspection PyProtectedMember
        used_threads = executor._max_workers
        lock = Lock()

        start_time = time()

        for _ in range(num_trials):
            futures = (executor.submit(lambda c: (c, run(**name_params(c))), config)
                       for config in product(*params.values()))
            # Need to collect all values to await execution.
            results.append(tuple(future.result() for future in as_completed(futures)))

        end_time = time()

    results = sorted((max(result, key=lambda t: t[1]) for result in results if len(result) > 0),
                     key=lambda t: t[1], reverse=True)
    median_case = results[0]  # len(results) // 2

    best_parameters, trn_acc = median_case
    best_parameters = name_params(best_parameters)

    # Test model with entire training set and best found parameters
    best_classifier = model(**best_parameters)
    best_classifier.fit(trn_ft, trn_tg)
    tst_acc = accuracy_score(tst_tg, best_classifier.predict(tst_ft))

    return OptimizeResult(name,
                          trn_acc,
                          tst_acc,
                          (end_time - start_time) / (len(results) * used_threads * num_trials),
                          best_parameters)


def feature_weights(alpha: float = 0) -> Dict[int, float]:
    """
    Adjusts feature weights to account for results from the binary set.
    :param alpha: how much to let the binary importances affect the weights.
    0 for balanced weights, 1 for only binary importances.
    :return: a dictionary mapping the classes to their weights.
    """

    return {c: 1 - alpha + w * alpha for c, w in enumerate(feature_importances_binary())}
