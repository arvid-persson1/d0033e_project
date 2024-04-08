from sklearn import model_selection, svm, preprocessing, metrics

from src.helpers import *

# TODO: remove this when missing values are handled properly.
from src import helpers
helpers.__df = helpers.__df.dropna()

features = get_numeric()
target = get_filtered(ids=True)

scaler = preprocessing.StandardScaler()
features = scaler.fit_transform(features)

# TODO: use given testing data instead of splitting training data
features, features_testing, target, target_testing = \
    model_selection.train_test_split(features, target, test_size=0.1)

# TODO: refactor all this

linear = svm.SVC(kernel='linear')
linear.fit(features, target.iloc[:, 0].tolist())
prediction_linear = linear.predict(features_testing)
accuracy_linear = metrics.accuracy_score(target_testing, prediction_linear)

quadratic = svm.SVC(kernel='poly', degree=2)
quadratic.fit(features, target.iloc[:, 0].tolist())
prediction_quadratic = quadratic.predict(features_testing)
accuracy_quadratic = metrics.accuracy_score(target_testing, prediction_quadratic)

cubic = svm.SVC(kernel='poly', degree=3)
cubic.fit(features, target.iloc[:, 0].tolist())
prediction_cubic = cubic.predict(features_testing)
accuracy_cubic = metrics.accuracy_score(target_testing, prediction_cubic)

quartic = svm.SVC(kernel='poly', degree=4)
quartic.fit(features, target.iloc[:, 0].tolist())
prediction_quartic = quartic.predict(features_testing)
accuracy_quartic = metrics.accuracy_score(target_testing, prediction_quartic)

rbf = svm.SVC(kernel='rbf')
rbf.fit(features, target.iloc[:, 0].tolist())
prediction_rbf = rbf.predict(features_testing)
accuracy_rbf = metrics.accuracy_score(target_testing, prediction_rbf)

sigmoid = svm.SVC(kernel='sigmoid')
sigmoid.fit(features, target.iloc[:, 0].tolist())
prediction_sigmoid = sigmoid.predict(features_testing)
accuracy_sigmoid = metrics.accuracy_score(target_testing, prediction_sigmoid)

print(f"{accuracy_linear=}")
print(f"{accuracy_quadratic=}")
print(f"{accuracy_cubic=}")
print(f"{accuracy_quartic=}")
print(f"{accuracy_rbf=}")
print(f"{accuracy_sigmoid=}")

# TAKEAWAYS
# linear seems to be the best
# without optimizing parameters, accuracy is around 80%
# results from varying c, gamma, coef0 and degree should be shown in the report
