from random import randint
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
from sklearn.base import ClassifierMixin
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score

from data import *
from joints import gesture_name, Joint
from optimize import feature_weights

# Override default environment setting; display plot in new window.
matplotlib.use("Qt5Agg")


BEST_FACTOR = 0.21521521521521522


def train_model(alpha: float = BEST_FACTOR) -> ClassifierMixin:
    """
    Trains the best available model.
    :param alpha: how much to let the binary importances affect the weights.
    0 for balanced weights, 1 for only binary importances.
    :return: a trained model; an instance of `ClassifierMixin`.
    """
    
    model = ExtraTreesClassifier(
        max_features=240,
        criterion="gini",
        max_depth=14,
        random_state=92,
        min_samples_split=1e-4,
        class_weight=feature_weights(alpha)
    )
    model.fit(training_features(), training_targets())

    return model


# noinspection PyUnresolvedReferences
def predictions(include_labels: bool = True, include_features: bool = False,
                alpha: float = BEST_FACTOR, copy: bool = False, sort: bool = False) -> DataFrame:
    """
    Gets the predicted classes for all gestures along with the actual classes, using the best available model.
    :param include_labels: whether to include the labels for the classes.
    :param include_features: whether to include the features.
    :param alpha: how much to let the binary importances affect the weights.
    0 for balanced weights, 1 for only binary importances.
    :param copy: whether to copy the features. Only used if `include_features` is enabled.
    :param sort: whether to sort the data.
    :return: A dataframe containing the predictions.
    """

    df = testing_features().copy(copy) if include_features else DataFrame()

    if include_labels:
        df["GESTURE_LABEL"] = testing_targets().apply(gesture_name)

    df["PREDICTION"] = train_model(alpha).predict(testing_features())
    df["ACTUAL"] = testing_targets()
    df["CORRECT"] = df["PREDICTION"] == df["ACTUAL"]

    if sort:
        df.sort_values("GESTURE_LABEL", inplace=True)

    return df


def predictions_incorrect(include_labels: bool = True, include_features: bool = False,
                          alpha: float = BEST_FACTOR, copy: bool = False, sort: bool = False) -> DataFrame:
    """
    Gets only the entries which the best available model classified incorrectly.
    :param include_labels: whether to include the labels for the classes.
    :param include_features: whether to include the features.
    :param alpha: how much to let the binary importances affect the weights.
    0 for balanced weights, 1 for only binary importances.
    :param copy: whether to copy the features. Only used if `include_features` is enabled.
    :param sort: whether to sort the data.
    :return: A dataframe containing the incorrect predictions.
    """

    df = predictions(include_labels, include_features, alpha, copy, sort)

    df = df[~df["CORRECT"]]
    df.drop(["CORRECT"], axis=1, inplace=True)

    return df


def plot_confusion(normalize: Any = None):
    """
    Plots the confusion matrix.
    :param normalize: how, if at all, to normalize the values. See `CunfusionMatrixDisplay.from_estimator`.
    """

    cm = ConfusionMatrixDisplay.from_estimator(
        train_model(BEST_FACTOR),
        testing_features(),
        testing_targets(),
        normalize=normalize
    )
    cm.plot()
    plt.show()


# noinspection PyUnresolvedReferences
def predict(count: int = 1, alpha: float = BEST_FACTOR, random_order: bool = True, visualize: bool = True):
    """
    Predicts the classes of unseen testing samples using the best available model.
    :param count: the number of samples to use.
    :param alpha: how much to let the binary importances affect the weights.
    0 for balanced weights, 1 for only binary importances.
    :param random_order: whether to present a random selection of samples in a random order.
    If this is disabled, starts from the first row.
    :param visualize: whether to display the result as a plot.
    If this is disabled, logs to stdout.
    """

    indices = (randint(0, 539) for _ in range(count)) if random_order else range(count)

    for index in indices:
        prediction = train_model(alpha).predict(testing_features().iloc[[index]])[0]
        target = testing_targets()[index]

        if visualize:
            testing = testing_data()
            xs = testing.iloc[index, range(0, 60, 3)]
            ys = testing.iloc[index, range(1, 60, 3)]
            zs = testing.iloc[index, range(2, 60, 3)]

            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.scatter(xs, ys, zs, color="dodgerblue")

            mid = (fig.subplotpars.left + fig.subplotpars.right) / 2
            plt.suptitle(f'"{gesture_name(prediction)}"', fontsize=16, x=mid)

            for i, j in Joint.connections():
                i -= 1
                j -= 1

                ax.plot(
                    [xs.iloc[i], xs.iloc[j]],
                    [ys.iloc[i], ys.iloc[j]],
                    [zs.iloc[i], zs.iloc[j]],
                    color="skyblue"
                )

            ax.view_init(elev=90, azim=-90)
            ax.set_axis_off()

            if prediction == target:
                plt.title("correct", color="green")
            else:
                plt.title(f'incorrect ("{gesture_name(target)}")', color="red")

            plt.show()

        else:
            if prediction == target:
                print(f"{index}\tcorrect")
            else:
                print(f'{index}\tincorrect (guessed "{gesture_name(prediction)}", was "{gesture_name(target)}")')


# noinspection PyUnresolvedReferences
def accuracy() -> float:
    """
    Calculates the accuracy of the best available model.
    :return: the accuracy; a value between 0 and 1 (inclusive).
    """

    return accuracy_score(testing_targets(), train_model().predict(testing_features()))


# noinspection PyUnresolvedReferences
def precision(average: str = "macro") -> float:
    """
    Calculates the precision of the best available model.
    :param average: the averaging method to use.
    Should be one of {"macro", "micro" and "weighted"}.
    :return: the precision; a value between 0 and 1 (inclusive).
    """

    return precision_score(testing_targets(), train_model().predict(testing_features()), average=average or "macro")


# noinspection PyUnresolvedReferences
def recall(average: str = "macro") -> float:
    """
    Calculates the precision of the best available model.
    :param average: the averaging method to use.
    Should be one of {"macro", "micro" and "weighted"}.
    :return: the precision; a value between 0 and 1 (inclusive).
    """

    return recall_score(testing_targets(), train_model().predict(testing_features()), average=average or "macro")


# noinspection PyUnresolvedReferences
def f1(average: str = "macro") -> float:
    """
    Calculates the F1-score of the best available model.
    :param average: the averaging method to use.
    Should be one of {"macro", "micro" and "weighted"}.
    :return: the F1-score; a value between 0 and 1 (inclusive).
    """

    return f1_score(testing_targets(), train_model().predict(testing_features()), average=average or "macro")
