from pandas import DataFrame
from typing import Iterable

from joints import Joint


def filter_df(
        df: DataFrame,
        joints: Iterable[Joint] | None = None,
        positions: bool | (bool, bool, bool) = False,
        angles: bool | (bool, bool, bool) = False,
        means: bool = False,
        deviations: bool = False,
        labels: bool = False
) -> DataFrame:
    """
    Selects only certain columns of the dataframe.
    :param df: the dataframe to filter.
    :param joints: which joints to include. All joints are included if this is `None`.
    :param positions: whether or not to include position data. A single boolean value applies to (x, y, z)
    while a triple of booleans allows for them to be specified separately.
    :param angles: whether or not to include angle data. A single boolean value applies to (1, 2, 3)
    while a triple of booleans allows for them to be specified separately.
    :param means: whether or not to include mean values.
    :param deviations: whether or not to include standard deviations.
    :param labels: whether or not to include gesture labels.
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
    if not joints:
        indices = set(range(240))
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

    return df.iloc[:, indices]
