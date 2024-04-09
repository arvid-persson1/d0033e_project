from typing import Callable, Dict, Any, Type

import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def accuracy(model: Type[ClassifierMixin],
             features: pd.DataFrame,
             target: pd.Series,
             test_split: float = 0.2,
             seed: int | None = None,
             preprocessor: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
             **params
             ) -> float:
    """
    Tests the accuracy of a model by using some of the training data for testing.
    Alias for splitting the data and calling `accuracy_from_tests`.
    :param model: the classifier to use.
    :param features: the features of the data.
    :param target: the target values.
    :param test_split: the share of the data to use for testing. Must be between 0 and 1.
    :param seed: the seed to use for data splitting. Uses a random seed if left blank.
    :param preprocessor: the preprocessing function to apply to the data before training, if any.
    :param params: parameters to pass to the model.
    :return: the accuracy on the testing data. Will always be between 0 and 1.
    """

    if test_split <= 0 or test_split >= 1:
        raise ValueError("test_split must be between 0 and 1")

    training_features, testing_features, training_target, testing_target = \
        train_test_split(features, target, test_size=test_split, random_state=seed)

    return accuracy_from_tests(model,
                               training_features,
                               training_target,
                               testing_features,
                               testing_target,
                               preprocessor,
                               **params)


# Python type annotations are not sophisticated enough to typecheck this function properly.
# noinspection PyUnresolvedReferences,PyArgumentList
def accuracy_from_tests(model: Type[ClassifierMixin],
                        training_features: pd.DataFrame,
                        training_target: pd.Series,
                        testing_features: pd.DataFrame,
                        testing_target: pd.Series,
                        preprocessor: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
                        **params
                        ) -> float:
    """
    Tests the accuracy of a model.
    :param model: the classifier to use.
    :param training_features: the features of the training data.
    :param training_target: the training target values.
    :param testing_features: the features of the testing data.
    :param testing_target: the testing target values.
    :param preprocessor: the preprocessing function to apply to the data before training, if any.
    :param params: parameters to pass to the model.
    :return: the accuracy on the testing data. Will always be between 0 and 1.
    """

    try:
        classifier = model(**params)
    except TypeError:
        raise TypeError("invalid parameters to model")

    if preprocessor is not None:
        training_features = preprocessor(training_features)

    classifier.fit(training_features, training_target)
    prediction = classifier.predict(testing_features)
    result = accuracy_score(testing_target, prediction)

    return result


def optimize_parameters(model: Type[ClassifierMixin],
                        # Is Any needed or are all parameters floats?
                        initial_guesses: Dict[str, float],
                        const_params: Dict[str, Any],
                        preprocessor: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
                        # TODO: provide options to fine-tune through arguments to minimize?
                        ) -> Dict[str, float]:



    if preprocessor is not None:
        training_features = preprocessor(training_features)

    # TODO: add callable for logging
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
