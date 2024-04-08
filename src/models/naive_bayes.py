from sklearn import model_selection, naive_bayes, metrics

from src.helpers import *

# TODO: remove this when missing values are handled properly.
from src import helpers
helpers.__df = helpers.__df.dropna()

features = get_numeric()
target = get_filtered(ids=True)

# TODO: use given testing data instead of splitting training data
features, features_testing, target, target_testing = \
    model_selection.train_test_split(features, target, test_size=0.1)

target = target.iloc[:, 0].tolist()
target_testing = target_testing.iloc[:, 0].tolist()

# TODO: refactor all this

gaussian = naive_bayes.GaussianNB()
gaussian.fit(features, target)
prediction_gaussian = gaussian.predict(features_testing)
accuracy_gaussian = metrics.accuracy_score(target_testing, prediction_gaussian)

bernoulli = naive_bayes.BernoulliNB()
bernoulli.fit(features, target)
prediction_bernoulli = bernoulli.predict(features_testing)
accuracy_bernoulli = metrics.accuracy_score(target_testing, prediction_bernoulli)

print(f"{accuracy_gaussian=}")
print(f"{accuracy_bernoulli=}")

# TAKEAWAYS
# naive bayes is efficient rather than accurate
# only gaussian and bernoulli can handle negative numbers
# bernoulli performs very poorly

# TODOS
# experiment with parameters for bernoulli
# https://scikit-learn.org/stable/modules/classes.html#module-sklearn.naive_bayes
