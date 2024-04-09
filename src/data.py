import pandas as pd
import numpy as np

from joints import Joint

__training = pd.read_csv("../data/training.csv", names=tuple(Joint.headers()))
__testing = pd.read_csv("../data/test.csv", names=tuple(Joint.headers()))
__rows, __cols = __training.shape


def __normalize(df: pd.DataFrame):
    # Spine x, y, z positions are on indices 30, 31, 32 respectively.
    df.iloc[:, np.arange(0, 60, 3)] -= df.iloc[:, 30].values.reshape(-1, 1)
    df.iloc[:, np.arange(1, 60, 3)] -= df.iloc[:, 31].values.reshape(-1, 1)
    df.iloc[:, np.arange(2, 60, 3)] -= df.iloc[:, 32].values.reshape(-1, 1)

    for index, row in df.iterrows():
        # Left and right shoulder positions are on indices 6-11.
        positions = df.iloc[index, 6:12].tolist()

        # Python thinks this code is unreachable.
        # noinspection PyUnreachableCode
        n = np.cross(positions[:3], positions[3:])

        # TODO: rotation normalization using n


__normalize(__training)
__normalize(__testing)

# FIXME: remove when missing data is handled properly
__training = __training.dropna()
__testing = __testing.dropna()

# Numeric values (positions and angles) are in columns 0-239,
# gesture ID's are in column 241.
__training_features = __training.iloc[:, range(240)]
__training_target = __training.iloc[:, 241]
__testing_features = __testing.iloc[:, range(240)]
__testing_target = __testing.iloc[:, 241]


# TODO: doc these?

def get_training_data() -> pd.DataFrame:
    return __training


def get_testing_data() -> pd.DataFrame:
    return __testing


def get_training_features() -> pd.DataFrame:
    return __training_features


def get_training_targets() -> pd.Series:
    return __training_target


def get_testing_features() -> pd.DataFrame:
    return __testing_features


def get_testing_targets() -> pd.Series:
    return __testing_target
