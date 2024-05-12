from random import randint
from sys import argv

from matplotlib import pyplot as plt
from sklearn import ensemble

from data import *
from joints import Joint, gesture_name

MODEL = ensemble.ExtraTreesClassifier(
    max_features=240,
    criterion="gini",
    max_depth=14,
    random_state=92,
    min_samples_split=1e-4
)
MODEL.fit(training_features(), training_targets())


def main():
    predict()


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
        prediction = MODEL.predict(testing_features().iloc[[index]])[0]
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


if __name__ == "__main__":
    main()
