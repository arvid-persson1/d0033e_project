from sklearn import model_selection, neighbors, metrics

from src.helpers import *

# TODO: remove this when missing values are handled properly.
from src import helpers
helpers.__training = helpers.__training.dropna()

features = get_numeric()
target = get_filtered(ids=True)

k = len(target.iloc[:, 0].unique())

# TODO: use given testing data instead of splitting training data
features, features_testing, target, target_testing = \
    model_selection.train_test_split(features, target, test_size=0.1)

target = target.iloc[:, 0].tolist()
target_testing = target_testing.iloc[:, 0].tolist()

# TODO: refactor all this

knn = neighbors.KNeighborsClassifier(n_neighbors=30)
knn.fit(features, target)
prediction_knn = knn.predict(features_testing)
accuracy_knn = metrics.accuracy_score(target_testing, prediction_knn)

radius = neighbors.RadiusNeighborsClassifier(radius=2.0)
radius.fit(features, target)
prediction_radius = radius.predict(features_testing)
accuracy_radius = metrics.accuracy_score(target_testing, prediction_radius)

centroid = neighbors.NearestCentroid()
centroid.fit(features, target)
prediction_centroid = centroid.predict(features_testing)
accuracy_centroid = metrics.accuracy_score(target_testing, prediction_centroid)

print(f"{accuracy_knn=}")
print(f"{accuracy_radius=}")
print(f"{accuracy_centroid=}")

# TAKEAWAYS
# considering the size of the training data, some gestures may not have many representatives
# here where the training data is split, should be remedied later
# very poor performance, though this is likely due to poor (or lack of) parameter tweaking

# TODOS
# https://scikit-learn.org/stable/modules/classes.html#module-sklearn.neighbors
