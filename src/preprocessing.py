import pandas as pd
from numpy import array, float64, cross, sin, cos, arccos, dot, append
from numpy.linalg import norm as mag

from joints import Joint

__training_raw = pd.read_csv("../data/training_filled.csv", names=tuple(Joint.headers()))
__testing_raw = pd.read_csv("../data/testing.csv", names=tuple(Joint.headers()))


def __process(row: pd.Series) -> pd.Series:
    # TODO: normalize rotation, then position

    cs = array(row.iloc[30:30 + 3], dtype=float64)
    ls = array(row.iloc[6:6 + 3], dtype=float64)
    rs = array(row.iloc[9:9 + 3], dtype=float64)

    # Normalize rotations

    cl = ls - cs
    cr = rs - cs
    # Python thinks this code is unreachable.
    # noinspection PyUnreachableCode
    n = 1 / (mag(cl) * mag(cr)) * cross(cl, cr)
    k = (0, 0, 1)

    # mag(k) is known to be 1
    theta = arccos(dot(n, k) / mag(n))

    st = sin(theta)
    ct = cos(theta)
    rotation = array([
        [ct,  0,  st,  0],
        [0,   1,  0,   0],
        [-st, 0,  ct,  0],
        [0,   0,  0,   1]
    ])

    for i in range(len(Joint)):
        row.iloc[i * 3: i * 3 + 3] = dot(rotation, append(row.iloc[i * 3: i * 3 + 3], 1))[:3]

    # Normalize positions

    row[0:60:3] -= cs[0]
    row[1:60:3] -= cs[1]
    row[2:60:3] -= cs[2]

    return row


__training_raw = __training_raw.apply(__process, axis=1)
__testing_raw = __testing_raw.apply(__process, axis=1)

__training_raw.to_csv("../data/training_processed.csv", index=False)
__testing_raw.to_csv("../data/testing_processed.csv", index=False)
