from typing import Any

import matplotlib
import matplotlib.pyplot as plt
from sklearn import *
from sklearn.metrics import ConfusionMatrixDisplay

from src.data import *
from src.joints import gesture_name

# Override default environment setting; display plot in new window.
matplotlib.use("Qt5Agg")

MODEL = ensemble.ExtraTreesClassifier(
    max_features=240,
    criterion="gini",
    max_depth=14,
    random_state=92,
    min_samples_split=1e-4
)
MODEL.fit(training_features(), training_targets())


def predictions(include_labels: bool = True, include_features: bool = False, copy: bool = False) -> DataFrame:
    """
    Gets the predicted classes for all gestures along with the actual classes, using the best available model.
    :param include_labels: whether to include the labels for the classes.
    :param include_features: whether to include the features.
    :param copy: whether to copy the features. Only used if `include_features` is enabled.
    :return: A dataframe containing the predictions.
    """

    df = testing_features().copy(copy) if include_features else DataFrame()

    if include_labels:
        df["GESTURE_LABEL"] = testing_targets().apply(gesture_name)

    df["PREDICTION"] = MODEL.predict(testing_features())
    df["ACTUAL"] = testing_targets()
    df["CORRECT"] = df["PREDICTION"] == df["ACTUAL"]

    return df


def predictions_incorrect(include_labels: bool = True, include_features: bool = False, copy: bool = False) -> DataFrame:
    """
    Gets only the entries which the best available model classified incorrectly.
    :param include_labels: whether to include the labels for the classes.
    :param include_features: whether to include the features.
    :param copy: whether to copy the features. Only used if `include_features` is enabled.
    :return: A dataframe containing the incorrect predictions.
    """

    df = predictions(include_labels, include_features, copy)

    df = df[~df["CORRECT"]]
    df.drop(["CORRECT"], axis=1, inplace=True)

    return df


def plot_confusion(normalize: Any = None):
    """
    Plots the confusion matrix.
    :param normalize: how, if at all, to normalize the values. See `CunfusionMatrixDisplay.from_estimator`.
    """

    cm = ConfusionMatrixDisplay.from_estimator(MODEL, testing_features(), testing_targets(), normalize=normalize)
    cm.plot()
    plt.show()
