from inspect import signature
from typing import Callable, List

from sklearn.base import ClusterMixin

from data import *

SEED = 1


def cluster(clusterer: Callable[..., ClusterMixin], df: str) -> List[int]:
    """
    Performs clustering with a model.
    :param clusterer: the constructor for the clusterer to use.
    :param df: the dataframe to perform clustering on.
    `training` for training set, `testing` for testing set.
    :return: the cluster labels. Guaranteed to be length 30.
    """

    try:
        if "random_state" in signature(clusterer).parameters:
            clusterer = clusterer(n_clusters=CLASSES, random_state=SEED)
        else:
            clusterer = clusterer(n_clusters=CLASSES)
    except TypeError:
        raise ValueError("Error: clusterer does not support fixed cluster count.")

    match df:
        case "training":
            df = training_features()
        case "testing":
            df = testing_features()
        case _:
            raise ValueError("Error: invalid df")

    return list(clusterer.fit_predict(df))
