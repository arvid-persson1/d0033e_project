from random import randint
from typing import Any

import matplotlib
from matplotlib import pyplot as plt
from pandas import DataFrame
from sklearn import *
from sklearn.metrics import ConfusionMatrixDisplay

from data import *
from joints import Joint

# Override default environment setting; display plot in new window.
matplotlib.use("Qt5Agg")

__model = ensemble.ExtraTreesClassifier(
    max_features=240,
    criterion="gini",
    max_depth=14,
    random_state=92,
    min_samples_split=1e-4
)
__model.fit(get_training_features(), get_training_targets())

__testing = get_testing_data()
__testing_features = get_testing_features()
__testing_targets = get_testing_targets()


def get_predictions(include_labels: bool = True, include_features: bool = False, copy: bool = False) -> DataFrame:
    """
    Gets the predicted classes for all gestures along with the actual classes, using the best available model.
    :param include_labels: whether to include the labels for the classes.
    :param include_features: whether to include the features.
    :param copy: whether to copy the features. Only used if `include_features` is enabled.
    :return: A dataframe containing the predictions.
    """

    df = __testing_features.copy(copy) if include_features else DataFrame()

    if include_labels:
        df["GESTURE_LABEL"] = __testing_targets.apply(__gesture_name)

    df["PREDICTION"] = __model.predict(__testing_features)
    df["ACTUAL"] = __testing_targets
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

    return ConfusionMatrixDisplay.from_estimator(__model, __testing_features, __testing_targets, normalize=normalize)


def predict(count: int = 5, random_order: bool = True, visualize: bool = True):
    if random_order:
        indices = (randint(0, 539) for _ in range(count))
    else:
        indices = range(count)

    for index in indices:
        prediction = __model.predict(__testing_features.iloc[[index]])[0]
        target = __testing_targets[index]

        if visualize:
            xs = __testing.iloc[index, range(0, 60, 3)]
            ys = __testing.iloc[index, range(1, 60, 3)]
            zs = __testing.iloc[index, range(2, 60, 3)]

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


def __gesture_name(gesture_id: int) -> str:
    match gesture_id:
        case 1:
            return "Afternoon"
        case 2:
            return "Baby"
        case 3:
            return "Big"
        case 4:
            return "Born"
        case 5:
            return "Bye"
        case 6:
            return "Calendar"
        case 7:
            return "Child"
        case 8:
            return "Cloud"
        case 9:
            return "Come"
        case 10:
            return "Daily"
        case 11:
            return "Dance"
        case 12:
            return "Dark"
        case 13:
            return "Day"
        case 14:
            return "Enjoy"
        case 15:
            return "Go"
        case 16:
            return "Hello"
        case 17:
            return "Home"
        case 18:
            return "Love"
        case 19:
            return "My"
        case 20:
            return "Name"
        case 21:
            return "No"
        case 22:
            return "Rain"
        case 23:
            return "Sorry"
        case 24:
            return "Strong"
        case 25:
            return "Study"
        case 26:
            return "Thank you"
        case 27:
            return "Welcome"
        case 28:
            return "Wind"
        case 29:
            return "Yes"
        case 30:
            return "You"

    raise ValueError("Invalid gesture ID.")
