import pandas as pd
import numpy as np
from typing import Iterable

from joints import Joint

__df = pd.read_csv("../data/training.csv", names=tuple(Joint.headers()))
__rows, __cols = __df.shape


def __normalize():
    global __df
    # spine x, y, z positions are on indices 30, 31, 32 respectively.
    __df.iloc[:, np.arange(0, 60, 3)] -= __df.iloc[:, 30].values.reshape(-1, 1)
    __df.iloc[:, np.arange(1, 60, 3)] -= __df.iloc[:, 31].values.reshape(-1, 1)
    __df.iloc[:, np.arange(2, 60, 3)] -= __df.iloc[:, 32].values.reshape(-1, 1)

    for index, row in __df.iterrows():
        # left and right shoulder positions are on indices 6-11
        positions = __df.iloc[index, 6:12].tolist()

        # Python thinks this code is unreachable.
        # noinspection PyUnreachableCode
        n = np.cross(positions[:3], positions[3:])

        # TODO: rotation normalization using n


__normalize()


def get_data() -> pd.DataFrame:
    return __df


def get_filtered(joints: Iterable[Joint] | None = None,
                 rows: Iterable[int] | int | None = None,
                 positions: bool | tuple[bool, bool, bool] = False,
                 angles: bool | tuple[bool, bool, bool] = False,
                 means: bool = False,
                 deviations: bool = False,
                 labels: bool = False,
                 ids: bool = False) -> pd.DataFrame:
    """
    Selects only certain columns of the dataframe.
    :param joints: which joints to include. All joints are included if this is `None`.
    :param rows: which row(s) to include, zero-indexed. All rows are included if this is `None`.
    :param positions: whether or not to include position data. A single boolean value applies to (x, y, z)
    while a triple of booleans allows for them to be specified separately.
    :param angles: whether or not to include angle data. A single boolean value applies to (1, 2, 3)
    while a triple of booleans allows for them to be specified separately.
    :param means: whether or not to include mean values.
    :param deviations: whether or not to include standard deviations.
    :param labels: whether or not to include gesture labels.
    :param ids: whether or not to incldue gesture ID's.
    :return: The dataframe with the filters applied.
    """

    if isinstance(positions, bool):
        pos_x = pos_y = pos_z = positions
    else:
        pos_x, pos_y, pos_z = positions

    if isinstance(angles, bool):
        angle_1 = angle_2 = angle_3 = angles
    else:
        angle_1, angle_2, angle_3 = angles

    indices = set()

    # add indices for selected joints
    if joints is None:
        indices = set(range(__cols - 2))
    else:
        for joint in joints:
            offset = (joint.value - 1) * 3
            indices |= set(range(offset, offset + 3))
            indices |= set(range(offset + 60, offset + 63))
            indices |= set(range(offset + 120, offset + 123))
            indices |= set(range(offset + 180, offset + 183))

    if not pos_x:
        indices -= set(range(0, 120, 3))
    if not pos_y:
        indices -= set(range(1, 120, 3))
    if not pos_z:
        indices -= set(range(2, 120, 3))
    if not angle_1:
        indices -= set(range(120, 240, 3))
    if not angle_2:
        indices -= set(range(121, 240, 3))
    if not angle_3:
        indices -= set(range(122, 240, 3))

    if not means:
        indices -= set(range(0, 60))
        indices -= set(range(120, 180))
    if not deviations:
        indices -= set(range(60, 120))
        indices -= set(range(180, 240))

    if labels:
        indices.add(240)
    if ids:
        indices.add(241)

    if rows is None:
        return __df.iloc[:, iter(indices)]
    else:
        return __df.iloc[rows, iter(indices)]
