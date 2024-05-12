from pandas import read_csv, DataFrame, Series

__training = read_csv("data/training_processed.csv")
__testing = read_csv("data/testing_processed.csv")

# Numeric values (positions and angles) are in columns 0-239,
# gesture ID's are in column 241.
__training_features = __training.iloc[:, range(240)]
__training_target = __training.iloc[:, 241]
__testing_features = __testing.iloc[:, range(240)]
__testing_target = __testing.iloc[:, 241]


def training_data() -> DataFrame:
    return __training


def testing_data() -> DataFrame:
    return __testing


def training_features() -> DataFrame:
    return __training_features


def training_targets() -> Series:
    return __training_target


def testing_features() -> DataFrame:
    return __testing_features


def testing_targets() -> Series:
    return __testing_target
