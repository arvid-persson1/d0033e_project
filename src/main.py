from training.tests import *


def main():
    dump(
        *test_svc(),
        *test_nu_svc(),
        *test_linear_model(),
        *test_naive_bayes(),
        *test_neighbors(),
        # Neural networks have been omitted as they are significantly more
        # expensive to optimize compared to other models. There will be
        # later iterations testing only neural networks.
        # *test_neural_network(),
        *test_tree()
    )


if __name__ == "__main__":
    main()
