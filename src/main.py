from functools import partial

from numpy import linspace
from sklearn import *

from optimize import *


def main():
    # visualize(0, True)

    # TODO: sklearn set_config, extract functions, reduce code repetition, optimize parameters
    # import models.svm
    # import models.naive_bayes
    # import models.neighbors
    # import models.decision_tree

    cfg, acc = optimize_parameters(
        partial(svm.SVC, kernel="linear"),
        lambda df: preprocessing.StandardScaler().fit_transform(df),
        C=linspace(0.001, 10, 10),
    )
    print(cfg, acc)
    pass


if __name__ == "__main__":
    main()
