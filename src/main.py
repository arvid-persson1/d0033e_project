import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import path

from joints import Joint

names = tuple(Joint.headers())

training = pd.read_csv("../data/training.csv", names=names)

#print(training.describe())
print(set(Joint))
