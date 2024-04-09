import matplotlib
from matplotlib import pyplot as plt

# override default environment setting; display plot in new window
matplotlib.use("Qt5Agg")

from data import *


def visualize(index: int = 0, axes: bool = False):
    """
    Visualizes a gesture as a 3d plot.
    :param index: the index of the gesture to visualize, 0-indexed.
    :param axes: whether or not to label the axes.
    """

    # Three filterings are definitely not necessary.
    xs = __training.iloc[index, range(0, 60, 3)]
    ys = __training.iloc[index, range(1, 60, 3)]
    zs = __training.iloc[index, range(2, 60, 3)]

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(xs, ys, zs)

    plt.title(f'"{__training.iloc[index, 240]}"')

    for i, j in Joint.connections():
        # adjust offset
        i -= 1
        j -= 1

        ax.plot(
            [xs.iloc[i], xs.iloc[j]],
            [ys.iloc[i], ys.iloc[j]],
            [zs.iloc[i], zs.iloc[j]],
            color="red"
        )

    if axes:
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

    ax.view_init(elev=-90, azim=90)

    plt.show()
