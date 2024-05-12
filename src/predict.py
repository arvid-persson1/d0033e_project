from random import randint
from typing import Any, Optional

import matplotlib
from matplotlib import pyplot as plt
from pandas import DataFrame
from sklearn import *
from sklearn.metrics import ConfusionMatrixDisplay

from data import *
from joints import Joint

# Override default environment setting; display plot in new window.
matplotlib.use("Qt5Agg")

MODEL = ensemble.ExtraTreesClassifier(
    max_features=240,
    criterion="gini",
    max_depth=14,
    random_state=92,
    min_samples_split=1e-4
)
MODEL.fit(get_training_features(), get_training_targets())


def get_predictions(include_labels: bool = True, include_features: bool = False, copy: bool = False) -> DataFrame:
    """
    Gets the predicted classes for all gestures along with the actual classes, using the best available model.
    :param include_labels: whether to include the labels for the classes.
    :param include_features: whether to include the features.
    :param copy: whether to copy the features. Only used if `include_features` is enabled.
    :return: A dataframe containing the predictions.
    """

    df = get_testing_features().copy(copy) if include_features else DataFrame()

    if include_labels:
        df["GESTURE_LABEL"] = get_testing_targets().apply(__gesture_name)

    df["PREDICTION"] = MODEL.predict(get_testing_features())
    df["ACTUAL"] = get_testing_targets()
    df["CORRECT"] = df["PREDICTION"] == df["ACTUAL"]

    return df


def get_incorrect(include_labels: bool = True, include_features: bool = False, copy: bool = False) -> DataFrame:
    """
    Gets only the entries which the best available model classified incorrectly.
    :param include_labels: whether to include the labels for the classes.
    :param include_features: whether to include the features.
    :param copy: whether to copy the features. Only used if `include_features` is enabled.
    :return: A dataframe containing the incorrect predictions.
    """

    df = get_predictions(include_labels, include_features, copy)

    df = df[~df["CORRECT"]]
    df.drop(["CORRECT"], axis=1, inplace=True)

    return df


def get_confusion(normalize: Any = None) -> ConfusionMatrixDisplay:
    """
    Gets the confusion matrix.
    :param normalize: how, if at all, to normalize the values. See `CunfusionMatrixDisplay.from_estimator`.
    :return: A confusion matrix ready to be displayed.
    """

    return ConfusionMatrixDisplay.from_estimator(MODEL, get_testing_features(), get_testing_targets(), normalize=normalize)


def predict(count: int = 1, random_order: bool = True, visualize: bool = True):
    """
    Predicts the classes of unseen testing samples using the best available model.
    :param count: the number of samples to use.
    :param random_order: whether to present a random selection of samples in a random order.
    If this is disabled, starts from the first row.
    :param visualize: whether to display the result as a plot.
    If this is disabled, logs to stdout.
    """

    indices = (randint(0, 539) for _ in range(count)) if random_order else range(count)

    for index in indices:
        prediction = MODEL.predict(get_testing_features().iloc[[index]])[0]
        target = get_testing_targets()[index]

        if visualize:
            testing = get_testing_data()
            xs = testing.iloc[index, range(0, 60, 3)]
            ys = testing.iloc[index, range(1, 60, 3)]
            zs = testing.iloc[index, range(2, 60, 3)]

            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.scatter(xs, ys, zs, color="dodgerblue")

            mid = (fig.subplotpars.left + fig.subplotpars.right) / 2
            plt.suptitle(f'"{__gesture_name(prediction)}"', fontsize=16, x=mid)

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
                plt.title(f'incorrect ("{__gesture_name(target)}")', color="red")

            plt.show()

        else:
            if prediction == target:
                print(f"{index}\tcorrect")
            else:
                print(f'{index}\tincorrect (guessed "{__gesture_name(prediction)}", was "{__gesture_name(target)}")')


__gesture_names = {
    1: "Afternoon",
    2: "Baby",
    3: "Big",
    4: "Born",
    5: "Bye",
    6: "Calendar",
    7: "Child",
    8: "Cloud",
    9: "Come",
    10: "Daily",
    11: "Dance",
    12: "Dark",
    13: "Day",
    14: "Enjoy",
    15: "Go",
    16: "Hello",
    17: "Home",
    18: "Love",
    19: "My",
    20: "Name",
    21: "No",
    22: "Rain",
    23: "Sorry",
    24: "Strong",
    25: "Study",
    26: "Thank you",
    27: "Welcome",
    28: "Wind",
    29: "Yes",
    30: "You"
}


def __gesture_name(gesture_id: int) -> Optional[str]:
    return __gesture_names.get(gesture_id)
