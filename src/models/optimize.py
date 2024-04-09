from typing import Callable, Dict, Any, Type

import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


class Optimizer:
    def __init__(self,
                 features: pd.DataFrame,
                 target: pd.Series,
                 test_split: float = 0.2,
                 seed: int | None = None):
        self.training_features, self.testing_features, self.training_target, self.testing_target = \
            train_test_split(features, target, test_size=test_split, random_state=seed)

    def test_accuracy(self,
                      # FIXME: incorrect or incomplete type annotation, creates unnecessary warnings
                      model: Type[ClassifierMixin],
                      preprocessor: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
                      **kwargs
                      ) -> float:
        training_features = self.training_features
        if preprocessor is not None:
            training_features = preprocessor(training_features)

        classifier = model(**kwargs)
        classifier.fit(training_features, self.training_target)
        prediction = classifier.predict(self.testing_features)
        accuracy = accuracy_score(self.testing_target, prediction)

        return accuracy

    def optimize_parameters(self,
                            model: Type[ClassifierMixin],
                            preprocessor: Callable[[pd.DataFrame], pd.DataFrame]
                            ) -> Dict[str, Any]:
        training_features = self.training_features
        if preprocessor is not None:
            training_features = preprocessor(training_features)

        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
