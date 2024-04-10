from training.tests import *


def main():
    dump(
        *test_svc(),
        *test_nu_svc(),
        *test_linear_model(),
        *test_naive_bayes(),
        *test_neighbors(),
        *test_neural_network(),
        *test_tree()
    )


if __name__ == "__main__":
    main()
