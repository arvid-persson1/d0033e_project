from pandas import value_counts

from data import *


def main():
    tr = value_counts(training_targets(), sort=False)
    tr.sort_index(inplace=True)
    tr.to_csv("tr.csv", index=False)
    print(tr)
    te = value_counts(testing_targets(), sort=False)
    te.sort_index(inplace=True)
    te.to_csv("te.csv", index=False)
    print(te)


if __name__ == "__main__":
    main()
