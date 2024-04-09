from sklearn import *

from optimize import *


def main():
    # visualize(0, True)

    # TODO: sklearn set_config, extract functions, reduce code repetition, optimize parameters
    # import models.svm
    # import models.naive_bayes
    # import models.neighbors
    # import models.decision_tree

    # print(accuracy(svm.SVC, lambda df: preprocessing.StandardScaler().fit_transform(df), kernel='linear'))
    r = optimize_parameters(
        svm.SVC,
        {"C": 0.001},
        kernel="linear"
    )

    print(r)
    pass


if __name__ == "__main__":
    main()
