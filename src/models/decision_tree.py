from sklearn import model_selection, tree, ensemble, metrics

from src.helpers import *

# TODO: remove this when missing values are handled properly.
from src import helpers
helpers.__training = helpers.__training.dropna()

features = get_numeric()
target = get_filtered(ids=True)

# TODO: use given testing data instead of splitting training data
features, features_testing, target, target_testing = \
    model_selection.train_test_split(features, target, test_size=0.1)

target = target.iloc[:, 0].tolist()
target_testing = target_testing.iloc[:, 0].tolist()

# TODO: refactor all this

decision = tree.DecisionTreeClassifier()
decision.fit(features, target)
prediction_decision = decision.predict(features_testing)
accuracy_decision = metrics.accuracy_score(target_testing, prediction_decision)

extra = tree.ExtraTreeClassifier()
extra.fit(features, target)
prediction_extra = extra.predict(features_testing)
accuracy_extra = metrics.accuracy_score(target_testing, prediction_extra)

# these should probably be in their own file or bundled with other ensemble models

random_forest = ensemble.RandomForestClassifier()
random_forest.fit(features, target)
prediction_random_forest = random_forest.predict(features_testing)
accuracy_random_forest = metrics.accuracy_score(target_testing, prediction_random_forest)

# according to the docs, HistGradientBoostingClassifier is much more performant for larger datasets,
# but both should be tested later
# gradient_boosting = ensemble.GradientBoostingClassifier()
gradient_boosting = ensemble.HistGradientBoostingClassifier()
gradient_boosting.fit(features, target)
prediction_gradient_boosting = gradient_boosting.predict(features_testing)
accuracy_gradient_boosting = metrics.accuracy_score(target_testing, prediction_gradient_boosting)

print(f"{accuracy_decision=}")
print(f"{accuracy_extra=}")
print(f"{accuracy_random_forest=}")
print(f"{accuracy_gradient_boosting=}")

# TAKEAWAYS
# random forest and gradient boosting show very good results even without parameter tweaking
# gradient boosting is very slow (hist variant is faster)

# TODOS
# experiment with seeds as well as normal parameters
# https://scikit-learn.org/stable/modules/classes.html#module-sklearn.tree
# https://scikit-learn.org/stable/modules/classes.html#module-sklearn.ensemble
