from helpers import *
from visualize import visualize

df = get_training()

print(df.iloc[0, 0])

# visualize(0, True)

# TODO: sklearn set_config, extract functions, reduce code repetition, optimize parameters
import models.svm
import models.naive_bayes
import models.neighbors
import models.decision_tree
