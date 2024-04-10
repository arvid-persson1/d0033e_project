from training.tests import *


def main():
    dump(
        *test_linear_model()
        # Neural networks have been omitted as they are significantly more
        # expensive to optimize compared to other models. There will be
        # later iterations testing only neural networks.
        # *test_neural_network(),
    )


if __name__ == "__main__":
    main()
