import numpy as np
import pandas as pd
from numpy.linalg import norm as mag
from scipy.spatial.transform import Rotation

from src.joints import Joint

__training_raw = pd.read_csv("../data/training.csv", names=tuple(Joint.headers()))
__testing_raw = pd.read_csv("../data/test.csv", names=tuple(Joint.headers()))


def __process(row: pd.Series) -> pd.Series:
    cs = np.array(row.iloc[30:30+3], dtype=np.float64)
    ls = np.array(row.iloc[6:6+3], dtype=np.float64)
    rs = np.array(row.iloc[9:9+3], dtype=np.float64)

    row[0:60:3] -= cs[0]
    row[1:60:3] -= cs[1]
    row[2:60:3] -= cs[2]

    cl = ls - cs
    cr = rs - cs
    # Python thinks this code is unreachable.
    # noinspection PyUnreachableCode
    n = 1 / (mag(cl) * mag(cr)) * np.cross(cl, cr)
    k = np.array((0, 0, 1))

    n_mag = mag(n)
    if n_mag != 0:
        theta = np.dot(n, k) / n_mag
    else:
        theta = 0

    rot = Rotation.from_rotvec((0, theta, 0))
    for i in range(len(Joint)):
        row.iloc[i:i+3] = rot.apply(row.iloc[i:i+3])

    return row


__training_raw = __training_raw.apply(__process, axis=1)
__testing_raw = __testing_raw.apply(__process, axis=1)

# This is not a good way to handle missing values.
__training_raw.fillna(__training_raw.iloc[:, :240].mean(), inplace=True)
__testing_raw.fillna(__testing_raw.iloc[:, :240].mean(), inplace=True)

__training_raw.to_csv("../data/training_processed.csv", index=False)
__testing_raw.to_csv("../data/testing_processed.csv", index=False)
