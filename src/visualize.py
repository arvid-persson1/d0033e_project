import matplotlib
from matplotlib import pyplot as plt

# override default environment setting; display plot in new window
# if this raises errors, try `pip install PyQt5`
matplotlib.use("Qt5Agg")

from helpers import *


def visualize(index: int = 0, axis: bool = False):
    """
    Visualizes a gesture as a 3d plot.
    :param index: the index of the gesture to visualize, 0-indexed.
    :param axis: whether or not to label the axis.
    """

    xs = get_filtered(rows=index, positions=(True, False, False), means=True)
    ys = get_filtered(rows=index, positions=(False, True, False), means=True)
    zs = get_filtered(rows=index, positions=(False, False, True), means=True)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(xs, ys, zs)

    plt.title(f'"{get_data().iloc[index, 240]}"')

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

    if axis:
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

    ax.view_init(elev=-90, azim=90)

    plt.show()
