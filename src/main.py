import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import path

from joints import Joint
from gestures import Gesture
from visualize import visualize

names = tuple(Joint.headers())

training = pd.read_csv("../data/training.csv", names=names)

visualize(training, 0)
