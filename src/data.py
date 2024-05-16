from pandas import read_csv, DataFrame, Series

__training = read_csv("./data/training_processed.csv")
__training_scaled = read_csv("./data/training_scaled.csv")
__training_pc = read_csv("./data/training_pc.csv")
__testing = read_csv("./data/testing_processed.csv")
__testing_scaled = read_csv("./data/testing_scaled.csv")
__testing_pc = read_csv("./data/testing_pc.csv")

# Numeric values (positions and angles) are in columns 0-239
__training_features = __training.iloc[:, 0:240]
__training_features_scaled = __testing_scaled.iloc[:, 0:240]
__training_target = __training["GESTURE_ID"]
__testing_features = __testing.iloc[:, 0:240]
__testing_features_scaled = __testing_scaled.iloc[:, 0:240]
__testing_target = __testing["GESTURE_ID"]


def training_data() -> DataFrame:
    return __training


def testing_data() -> DataFrame:
    return __testing


def training_features() -> DataFrame:
    return __training_features


def training_features_scaled() -> DataFrame:
    return __training_features_scaled


def training_pc() -> DataFrame:
    return __training_pc


def training_targets() -> Series:
    return __training_target


def testing_features() -> DataFrame:
    return __testing_features


def testing_features_scaled() -> DataFrame:
    return __testing_features_scaled


def testing_pc() -> DataFrame:
    return __testing_pc


def testing_targets() -> Series:
    return __testing_target
