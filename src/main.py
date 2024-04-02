import pandas as pd

from joints import Joint
from visualize import visualize

names = tuple(Joint.headers())

training = pd.read_csv("../data/training.csv", names=names)

visualize(training, 100)
