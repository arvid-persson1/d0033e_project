from typing import Callable, Dict, Any, Type

import sklearn
from scipy.optimize import minimize
from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score

from src.data import *

sklearn.set_config(assume_finite=True)


def accuracy(model: Type[ClassifierMixin],
             preprocessor: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
             **params
             ) -> float:
    """
    Tests the accuracy of a model.
    :param model: the classifier to use.
    :param preprocessor: the preprocessing function to apply to the data before training, if any.
    :param params: parameters to pass to the model.
    :return: the accuracy on the testing data. Will always be between 0 and 1.
    """

    training_features = get_training_features()
    training_targets = get_training_targets()
    testing_features = get_testing_features()
    testing_targets = get_testing_targets()

    if preprocessor is not None:
        training_features = preprocessor(training_features)
        testing_features = preprocessor(testing_features)

    return __from_tests(model,
                        training_features,
                        training_targets,
                        testing_features,
                        testing_targets,
                        **params)


# Python type annotations are not sophisticated enough to typecheck this function properly.
# noinspection PyUnresolvedReferences,PyArgumentList
def __from_tests(model: Type[ClassifierMixin],
                 training_features: pd.DataFrame,
                 training_targets: pd.Series,
                 testing_features: pd.DataFrame,
                 testing_targets: pd.Series,
                 **params
                 ) -> float:
    try:
        classifier = model(**params)
    except TypeError:
        raise TypeError("invalid parameters to model")

    classifier.fit(training_features, training_targets)
    prediction = classifier.predict(testing_features)
    result = accuracy_score(testing_targets, prediction)

    return result


def optimize_parameters(model: Type[ClassifierMixin],
                        # Is Any needed or are all parameters floats?
                        initial_guesses: Dict[str, float],
                        preprocessor: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
                        # TODO: provide options to fine-tune through arguments to minimize?
                        **params
                        ) -> Dict[str, float]:
    training_features = get_training_features()
    training_targets = get_training_targets()
    testing_features = get_testing_features()
    testing_targets = get_testing_targets()

    if preprocessor is not None:
        training_features = preprocessor(training_features)
        testing_features = preprocessor(testing_features)

    def fun(x):
        acc = __from_tests(model,
                           training_features,
                           training_targets,
                           testing_features,
                           testing_targets,
                           **__bundle(names, x),
                           **params)

        # We're maximizing.
        print(acc, __bundle(names, x))
        return -acc

    names = list(initial_guesses.keys())
    x0 = np.array(tuple(initial_guesses.values()))

    result = minimize(fun, x0)

    if result.success:
        return __bundle(names, result.x)
    else:
        raise ValueError(f"Error: {result.message}")

    # TODO: add callable for logging
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html


def __bundle(names: Iterable[str], values: Iterable[float]) -> Dict[str, float]:
    return dict(arg for arg in zip(names, values))
