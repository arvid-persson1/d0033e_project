import pandas as pd

__training = pd.read_csv("../data/training_processed.csv.csv")
__testing = pd.read_csv("../data/testing_processed.csv.csv")

# FIXME: remove when missing data is handled properly
# __training = __training.dropna()
# __testing = __testing.dropna()

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
